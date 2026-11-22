"""Tests for server/segmentot_server.py. Needs torch, numpy, opencv and rasterio; loads no model.

  uv run --no-project --with torch --with numpy --with opencv-python-headless \
      --with rasterio python -m unittest tests/test_server.py -v
"""
import hmac
import http.client
import importlib.util
import json
import os
import sys
import threading
import types
import unittest
from unittest import mock

import numpy as np
import torch

_spec = importlib.util.spec_from_file_location(
    "segmentot_server", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server", "segmentot_server.py"))
srv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(srv)


class Clean(unittest.TestCase):
    def setUp(self):
        self._s = dict(srv.S)
        srv.S["model_id"] = "facebook/sam2.1-hiera-tiny"
        srv._emb.clear()
        srv._crop_emb.clear()
        srv._masks.clear()

    def tearDown(self):
        srv.S.clear()
        srv.S.update(self._s)
        srv._emb.clear()
        srv._crop_emb.clear()
        srv._masks.clear()


class Polygons(unittest.TestCase):
    def areas(self, mask, outline):
        return [[srv._shoelace(r) for r in rings] for rings in srv.mask_to_polygons(mask, 4, outline)]

    def test_squares_exact_area_both_outlines(self):
        for n in (10, 20):
            m = np.zeros((40, 40), bool)
            m[5:5 + n, 7:7 + n] = True
            for outline in ("pixel", "smooth"):
                self.assertEqual(self.areas(m, outline), [[n * n]], (n, outline))

    def test_one_pixel_strip_kept(self):
        m = np.zeros((20, 120), bool)
        m[10, 0:100] = True
        self.assertEqual(self.areas(m, "pixel"), [[100]])
        self.assertEqual(self.areas(m, "smooth"), [[100]])

    def test_hole_kept(self):
        m = np.zeros((40, 40), bool)
        m[2:32, 2:32] = True
        m[12:22, 12:22] = False
        for outline in ("pixel", "smooth"):
            self.assertEqual(self.areas(m, outline), [[900, 100]], outline)

    def test_rings_closed(self):
        m = np.zeros((40, 40), bool)
        m[5:15, 5:15] = True
        for outline in ("pixel", "smooth"):
            ring = srv.mask_to_polygons(m, 4, outline)[0][0]
            self.assertEqual(ring[0], ring[-1], outline)

    def test_smooth_does_not_shear_rounded_corner(self):
        m = np.zeros((40, 40), bool)
        m[5:25, 5:25] = True
        m[5, 5] = False                                 # 1-px rounded corner on a 20 x 20 square
        ring = srv.mask_to_polygons(m, 4, "smooth")[0][0]
        self.assertEqual(ring[0], ring[-1])
        self.assertEqual(srv._shoelace(ring), 399.0)    # 1.0 eps on closed ring sheared the whole top edge

    def test_logits_to_polygons_subpixel_smooth(self):
        # Continuous signed-distance logit field for a circle of radius 10.25 centred at (20.25, 20.5)
        yy, xx = np.mgrid[0:48, 0:48]
        logits = (10.25 - np.hypot(xx + 0.5 - 20.25, yy + 0.5 - 20.5)).astype(np.float32)
        valid = np.ones((48, 48), bool)
        polys = srv.logits_to_polygons(logits, valid, 4)
        self.assertEqual(len(polys), 1)
        ring = polys[0][0]
        self.assertEqual(ring[0], ring[-1])
        # Sub-pixel vertices (fractional coordinates, not all snapped to integers)
        self.assertTrue(any(abs(x - round(x)) > 1e-3 or abs(y - round(y)) > 1e-3 for x, y in ring))
        truth_area = np.pi * 10.25 ** 2
        self.assertLess(abs(srv._shoelace(ring) - truth_area) / truth_area, 0.03)
        # No-data clipping cuts the right half cleanly at x = 20
        valid[:, 20:] = False
        half = srv.logits_to_polygons(logits, valid, 4)
        self.assertEqual(len(half), 1)
        self.assertLessEqual(max(x for x, _ in half[0][0]), 20.0 + 1e-4)


# --------------------------------------------------------------- fakes for the GPU -> CPU fallback
def _gpu_check():
    if srv.S["device"] != "cpu":
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB.\nmore detail")


class FakeModel:
    config = types.SimpleNamespace(prompt_encoder_config=types.SimpleNamespace(image_size=1024))
    dtype = torch.float32

    def __init__(self):
        self.moved, self.calls = [], []

    def to(self, dev, dtype=None):
        self.moved.append(dev)
        self.dtype = dtype or self.dtype
        return self

    def float(self):
        self.dtype = torch.float32
        return self

    def get_image_embeddings(self, pv):
        _gpu_check()
        return torch.zeros(1)

    def __call__(self, **kw):
        _gpu_check()
        self.calls.append(kw)
        return types.SimpleNamespace(pred_masks=torch.zeros(1, 1, 1, 4, 4), iou_scores=torch.tensor([[[0.9]]]))


class FakeProc:
    def __call__(self, images, return_tensors):
        return {"pixel_values": torch.zeros(1, 3, 4, 4)}

    def post_process_masks(self, masks, sizes):
        h, w = sizes[0].tolist()
        m = torch.zeros(1, 1, h, w)
        m[0, 0, 2:12, 2:12] = 1                    # a 10 x 10 px object
        return [m]


def opaque(n=16):
    return np.full((n, n, 4), 255, np.uint8).tobytes()


# "meta" stands in for the GPU: every torch build has it, so the real .to(dev) calls work here
class Fallback(Clean):
    def setUp(self):
        super().setUp()
        srv.S.update(model=FakeModel(), proc=FakeProc(), device="meta", notice=None)

    def test_cached_image_gpu_fails_then_need_image(self):
        srv._emb["a"] = (torch.zeros(1), (16, 16), np.ones((16, 16), bool))
        res = srv.predict({"image_id": "a", "points": [[5, 5]], "labels": [1]})
        self.assertEqual(res, {"need_image": True})
        self.assertEqual(srv.S["device"], "cpu")
        self.assertEqual(len(srv._emb), 0)
        self.assertEqual(srv.S["model"].moved, ["cpu"])
        self.assertEqual(srv.S["notice"], "GPU failed (CUDA out of memory. Tried to allocate 2.00 GiB.); "
                                          "now running on the CPU (slower).")

    def test_gpu_fails_with_image_retry_succeeds(self):
        res = srv.predict({"image_id": "b", "size": [16, 16], "points": [[5, 5]], "labels": [1],
                           "outline": "pixel", "min_area_px": 4}, opaque())
        self.assertEqual(srv.S["device"], "cpu")
        self.assertEqual([[srv._shoelace(r) for r in rings] for c in res["candidates"] for rings in c["polygons"]],
                         [[100]])
        self.assertIn("b", srv._emb)

    def test_alpha_0_is_no_data(self):
        srv.S["device"] = "cpu"
        a = np.full((16, 20, 4), 255, np.uint8)        # 20 wide, 16 high: rows of 20 pixels
        a[:, 7:, 3] = 0                                # the 10 x 10 object loses its columns 7 and up
        res = srv.predict({"image_id": "f", "size": [20, 16], "points": [[5, 5]], "labels": [1]}, a.tobytes())
        self.assertEqual([srv._shoelace(c["polygons"][0][0]) for c in res["candidates"]], [50])

    def test_no_fallback_for_bad_request(self):
        for req, px in (({"labels": [1, 0]}, opaque()),
                        ({"size": [16, 16]}, opaque()[:-1]),                   # not w * h * 4 bytes
                        ({"size": [5000, 4000]}, bytes(5000 * 4000 * 4))):     # over MAX_PIXELS
            with self.assertRaises(ValueError):
                srv.predict(dict({"image_id": "c", "points": [[5, 5]], "labels": [1]}, **req), px)
        self.assertEqual(srv.S["device"], "meta")
        self.assertIsNone(srv.S["notice"])

    def test_use_cpu_frees_cuda_cache(self):
        srv.S["device"] = "cuda"
        srv._emb["a"] = (torch.zeros(1), (16, 16), np.ones((16, 16), bool))
        with mock.patch.object(srv.torch.cuda, "empty_cache") as empty:
            srv.use_cpu(RuntimeError("CUDA out of memory"))
        empty.assert_called_once()
        self.assertEqual((srv.S["device"], len(srv._emb), srv.S["model"].moved), ("cpu", 0, ["cpu"]))

    def test_cpu_failure_is_plain_error(self):
        srv.S["device"] = "cpu"
        srv.S["model"].get_image_embeddings = mock.Mock(side_effect=RuntimeError("boom"))
        with self.assertRaises(RuntimeError):
            srv.predict({"image_id": "d", "size": [16, 16], "points": [[5, 5]], "labels": [1]}, opaque())
        self.assertIsNone(srv.S["notice"])

    def test_each_model_family_has_its_loader(self):
        self.assertEqual(srv.family("facebook/sam2.1-hiera-tiny"), ("Sam2Model", "Sam2Processor"))
        self.assertEqual(srv.family("yonigozlan/EdgeTAM-hf"), ("EdgeTamModel", "Sam2Processor"))
        self.assertEqual(srv.family("facebook/sam3"), ("Sam3TrackerModel", "Sam3TrackerProcessor"))

    def load(self, model, device):
        model.eval = lambda: model
        tf = types.SimpleNamespace(
            Sam2Model=types.SimpleNamespace(from_pretrained=lambda _: model),
            Sam2Processor=types.SimpleNamespace(from_pretrained=lambda _: FakeProc()),
            utils=types.SimpleNamespace(logging=types.SimpleNamespace(set_verbosity_error=lambda: None)))
        srv.S.update(model=None, proc=None, device=device)
        with mock.patch.dict(sys.modules, {"transformers": tf}), \
                mock.patch.object(srv.torch.cuda, "empty_cache"):
            srv.load()
        self.assertIs(srv.S["model"], model)

    def test_load_falls_back_when_move_to_gpu_fails(self):
        model = FakeModel()
        model.to = mock.Mock(side_effect=[RuntimeError("CUDA error: no kernel image")] * 2 + [model])
        self.load(model, "cuda")
        self.assertEqual([c.args for c in model.to.call_args_list],              # float16, then float32, then the CPU
                         [("cuda", torch.float16), ("cuda", torch.float32), ("cpu", torch.float32)])
        self.assertEqual(srv.S["device"], "cpu")
        self.assertTrue(srv.S["notice"].startswith("GPU failed (CUDA error: no kernel image)"))

    def test_load_falls_back_when_warm_up_fails(self):
        self.load(FakeModel(), "meta")                 # the fake fails every call off the CPU
        model = srv.S["model"]
        self.assertEqual((srv.S["device"], model.moved, model.dtype), ("cpu", ["meta", "meta", "cpu"], torch.float32))
        self.assertTrue(srv.S["notice"].startswith("GPU failed (CUDA out of memory."))
        self.assertEqual(len(model.calls), 0)                           # no warm-up on the CPU

    def test_float16_nan_goes_back_to_float32_on_the_gpu(self):
        class NanInHalf(FakeModel):                    # GTX 16xx
            def get_image_embeddings(self, pv):
                return torch.zeros(1)

            def __call__(self, **kw):
                self.calls.append(kw)
                v = float("nan") if self.dtype == torch.float16 else 0.0
                return types.SimpleNamespace(pred_masks=torch.full((1, 1, 3, 4, 4), v), iou_scores=torch.zeros(1, 1, 3))
        self.load(NanInHalf(), "meta")
        self.assertEqual((srv.S["device"], srv.S["model"].dtype, srv.S["notice"]), ("meta", torch.float32, None))
        self.assertEqual(len(srv.S["model"].calls), 2)


# --------------------------------------------------------------- the real HTTP server
class Http(Clean):
    @classmethod
    def setUpClass(cls):
        cls.server = srv.Server(("127.0.0.1", 0), srv.Handler)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def ask(self, method, path, headers=None, body=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request(method, path, body=body, headers=headers or {})
        r = c.getresponse()
        out = r.status, json.loads(r.read())
        c.close()
        return out

    def test_no_header_forbidden(self):
        self.assertEqual(self.ask("GET", "/health")[0], 403)
        self.assertEqual(self.ask("POST", "/predict", body=b"{}")[0], 403)

    def test_foreign_host_forbidden(self):
        h = {"X-Segmentot": "1", "Host": f"evil.example:{self.port}"}
        self.assertEqual(self.ask("GET", "/health", h)[0], 403)

    def test_oversized_body_413(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.putrequest("POST", "/predict")
        c.putheader("X-Segmentot", "1")
        c.putheader("Content-Length", str(srv.MAX_BODY + 1))
        c.endheaders()
        self.assertEqual(c.getresponse().status, 413)
        c.close()

    def test_health_ok(self):
        for host in (f"127.0.0.1:{self.port}", f"localhost:{self.port}"):
            code, r = self.ask("GET", "/health", {"X-Segmentot": "1", "Host": host})
            self.assertEqual(code, 200)
            self.assertTrue(r["ok"])
            self.assertFalse(r["loaded"])
            self.assertEqual(r["device"], srv.S["device"])
            self.assertEqual(r["image_size"], 1024)
            self.assertNotIn("notice", r)

    def test_proves_itself_with_the_key(self):
        def proof(headers):
            c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
            c.request("GET", "/health", headers=headers)
            r = c.getresponse()
            r.read()
            c.close()
            return r.getheader("X-Segmentot-Proof")
        want = hmac.new(b"secret", f"{self.port}:abc".encode(), "sha256").hexdigest()
        with mock.patch.object(srv, "KEY", b"secret"):
            self.assertEqual(proof({"X-Segmentot": "1", "X-Segmentot-Nonce": "abc"}), want)
            self.assertIsNone(proof({"X-Segmentot": "1"}))
        self.assertIsNone(proof({"X-Segmentot": "1", "X-Segmentot-Nonce": "abc"}))       # started without a key

    def pixels_header(self, size=(16, 16)):
        return {"X-Segmentot": "1", "Content-Type": "application/octet-stream", "X-Segmentot-Request": json.dumps(
            {"image_id": "e", "size": size, "points": [[5, 5]], "labels": [1]}, separators=(",", ":"))}

    def test_predict_json_without_image_and_bad_pixels(self):
        srv.S.update(model=FakeModel(), proc=FakeProc(), device="meta", notice=None)
        body = json.dumps({"image_id": "e", "points": [[5, 5]], "labels": [1]})
        code, r = self.ask("POST", "/predict", {"X-Segmentot": "1"}, body)
        self.assertEqual((code, r["need_image"]), (200, True))
        code, r = self.ask("POST", "/predict", self.pixels_header(), opaque()[:-4])
        self.assertEqual((code, srv.S["device"]), (500, "meta"))        # an error, not a GPU failure
        self.assertIn("4 bytes (RGBA)", r["error"])

    def test_predict_reports_fallback(self):
        srv.S.update(model=FakeModel(), proc=FakeProc(), device="meta", notice=None)
        code, r = self.ask("POST", "/predict", self.pixels_header(), opaque())
        self.assertEqual(code, 200)
        self.assertEqual(len(r["candidates"]), 1)
        self.assertIn("mask_id", r["candidates"][0])
        self.assertEqual(r["image_size"], 1024)
        self.assertEqual(r["device"], "cpu")
        self.assertTrue(r["notice"].startswith("GPU failed ("))
        self.assertIn("seconds", r)
        code, r = self.ask("GET", "/health", {"X-Segmentot": "1"})
        self.assertEqual((code, r["device"], r["notice"][:11]), (200, "cpu", "GPU failed "))


# --------------------------------------------------------------- accuracy fixes
class AccuracyFixes(Clean):
    def test_keep_clicked_respects_holes(self):
        m = np.zeros((40, 40), bool)
        m[2:38, 2:38] = True
        m[10:30, 10:30] = False                        # hole spanning [10, 30]
        m[16:24, 16:24] = True                         # inner island inside the hole
        polys = srv.mask_to_polygons(m, 4, "pixel")
        self.assertEqual(len(polys), 2)
        # Click inside the inner island (also inside the donut's hole): must keep ONLY the island
        kept = srv.keep_clicked(polys, np.array([[20.0, 20.0]]), np.array([1]))
        self.assertEqual(len(kept), 1)
        self.assertEqual(srv._shoelace(kept[0][0]), 64.0)
        # Donut alone: click inside its hole has positive signed distance (> 0)
        donut = [p for p in polys if len(p) > 1][0]
        self.assertGreater(srv._poly_dist(donut, 12.0, 12.0), 0.0)
        self.assertLess(srv._poly_dist(donut, 5.0, 5.0), 0.0)

    def test_covers_ranked_after_min_area_filter(self):
        class TwoCandModel(FakeModel):
            def __call__(self, **kw):
                self.calls.append(kw)
                return types.SimpleNamespace(pred_masks=torch.zeros(1, 1, 2, 4, 4),
                                             iou_scores=torch.tensor([[[0.95, 0.80]]]))

        class TwoCandProc(FakeProc):
            def post_process_masks(self, masks, sizes):
                h, w = sizes[0].tolist()
                m = torch.zeros(1, 2, h, w)
                # cand 0 (score 0.95): 1-px speck at click (5, 5) (< min_area 4) + unrelated distant blob
                m[0, 0, 5, 5] = 1
                m[0, 0, 40:55, 40:55] = 1
                # cand 1 (score 0.80): real 10x10 object covering click (5, 5)
                m[0, 1, 2:12, 2:12] = 1
                return [m]

        srv.S.update(model=TwoCandModel(), proc=TwoCandProc(), device="cpu", notice=None)
        res = srv.predict({"image_id": "rank", "size": [64, 64], "points": [[5.5, 5.5]], "labels": [1],
                           "outline": "pixel", "min_area_px": 4}, opaque(64))
        self.assertEqual(len(res["candidates"]), 2)
        self.assertEqual(res["candidates"][0]["score"], 0.8)
        self.assertEqual(srv._shoelace(res["candidates"][0]["polygons"][0][0]), 100.0)

    def test_mask_id_cached_forwarded_and_float32_points(self):
        class HalfModel(FakeModel):
            dtype = torch.float16

            def __call__(self, **kw):
                self.calls.append(kw)
                n = 3 if kw.get("multimask_output") else 1
                pm = torch.arange(1, n + 1, dtype=torch.float16).view(1, 1, n, 1, 1).expand(1, 1, n, 4, 4).clone()
                sc = torch.linspace(0.7, 0.9, n).view(1, 1, n)
                return types.SimpleNamespace(pred_masks=pm, iou_scores=sc)

        class MultiProc(FakeProc):
            def post_process_masks(self, masks, sizes):
                h, w = sizes[0].tolist()
                n = masks.shape[2]
                m = torch.zeros(1, n, h, w)
                m[0, :, 2:12, 2:12] = 1
                return [m]

        model = HalfModel()
        srv.S.update(model=model, proc=MultiProc(), device="cpu", notice=None)
        # Warm-up uses float32 input_points even on float16 model
        self.assertTrue(srv.warm_up())
        self.assertEqual(model.calls[-1]["input_points"].dtype, torch.float32)

        # Click 1: single click -> multimask_output=True, no input_masks, float32 input_points
        r1 = srv.predict({"image_id": "m1", "size": [16, 16], "points": [[5.0, 5.0]], "labels": [1]}, opaque())
        c1 = model.calls[-1]
        self.assertTrue(c1["multimask_output"])
        self.assertNotIn("input_masks", c1)
        self.assertEqual(c1["input_points"].dtype, torch.float32)
        self.assertEqual(len(r1["candidates"]), 3)
        mid = r1["candidates"][0]["mask_id"]
        self.assertIn(mid, srv._masks)
        self.assertEqual(srv._masks[mid][1].shape, (1, 1, 4, 4))
        self.assertEqual(srv._masks[mid][1].dtype, torch.float32)

        # Click 2: multi-click with mask_id -> forwards input_masks in model.dtype, float32 input_points
        r2 = srv.predict({"image_id": "m1", "points": [[5.0, 5.0], [14.0, 14.0]], "labels": [1, 0], "mask_id": mid})
        c2 = model.calls[-1]
        self.assertFalse(c2["multimask_output"])
        self.assertIn("input_masks", c2)
        self.assertEqual(c2["input_masks"].dtype, torch.float16)
        self.assertEqual(c2["input_masks"].shape, (1, 1, 4, 4))
        self.assertTrue(torch.allclose(c2["input_masks"].float(), torch.full((1, 1, 4, 4), 3.0)))
        self.assertEqual(c2["input_points"].dtype, torch.float32)
        self.assertEqual(len(r2["candidates"]), 1)

        # Single click ignores any stale mask_id
        srv.predict({"image_id": "m1", "points": [[5.0, 5.0]], "labels": [1], "mask_id": mid})
        self.assertNotIn("input_masks", model.calls[-1])

        # Multi-click with mask_id from another image_id ignores input_masks
        srv.predict({"image_id": "m2", "size": [16, 16], "points": [[5.0, 5.0], [14.0, 14.0]],
                     "labels": [1, 0], "mask_id": mid}, opaque())
        self.assertNotIn("input_masks", model.calls[-1])

        # LRU eviction caps _masks at CACHE_MASKS
        for i in range(srv.CACHE_MASKS + 10):
            srv._put_mask("m1", torch.full((1, 1, 4, 4), float(i)))
        self.assertEqual(len(srv._masks), srv.CACHE_MASKS)

    def test_low_score_input_mask_falls_back_to_unconditioned(self):
        class CollapseWithMaskModel(FakeModel):
            def __call__(self, **kw):
                self.calls.append(kw)
                sc = 0.0 if "input_masks" in kw else 0.85
                return types.SimpleNamespace(pred_masks=torch.zeros(1, 1, 1, 4, 4),
                                             iou_scores=torch.tensor([[[sc]]]))

        model = CollapseWithMaskModel()
        srv.S.update(model=model, proc=FakeProc(), device="cpu", notice=None)
        r1 = srv.predict({"image_id": "fb", "size": [16, 16], "points": [[5.0, 5.0]], "labels": [1]}, opaque())
        mid = r1["candidates"][0]["mask_id"]
        model.calls.clear()
        r2 = srv.predict({"image_id": "fb", "points": [[5.0, 5.0], [14.0, 14.0]], "labels": [1, 0], "mask_id": mid})
        self.assertEqual(len(model.calls), 2)
        self.assertIn("input_masks", model.calls[0])
        self.assertNotIn("input_masks", model.calls[1])
        self.assertEqual(r2["candidates"][0]["score"], 0.85)

    def test_small_raster_under_1pct_valid_area(self):
        srv.S.update(model=FakeModel(), proc=FakeProc(), device="cpu", notice=None)
        # Completely empty -> always rejected
        empty = np.zeros((128, 128, 4), np.uint8).tobytes()
        r0 = srv.predict({"image_id": "e0", "size": [128, 128], "points": [[10, 10]], "labels": [1]}, empty)
        self.assertEqual(r0, {"candidates": [], "error": "No imagery here"})
        self.assertNotIn("e0", srv._emb)

        # 8x8 valid patch in 128x128 = 0.39% (< 1%)
        a = np.zeros((128, 128, 4), np.uint8)
        a[2:10, 2:10] = 255
        px = a.tobytes()
        # Click far from valid patch -> rejected, not cached
        r_far = srv.predict({"image_id": "small", "size": [128, 128], "points": [[100, 100]], "labels": [1]}, px)
        self.assertEqual(r_far, {"candidates": [], "error": "No imagery here"})
        self.assertNotIn("small", srv._emb)
        # Click on/near valid patch -> accepted and cached
        r_near = srv.predict({"image_id": "small", "size": [128, 128], "points": [[6, 6]], "labels": [1]}, px)
        self.assertIn("small", srv._emb)
        self.assertEqual(len(r_near["candidates"]), 1)
        self.assertEqual(srv._shoelace(r_near["candidates"][0]["polygons"][0][0]), 64.0)

    def test_model_image_size(self):
        srv.S["model"] = None
        srv.S["model_id"] = "facebook/sam2.1-hiera-base-plus"
        self.assertEqual(srv.model_image_size(), 1024)
        srv.S["model_id"] = "facebook/sam3"
        self.assertEqual(srv.model_image_size(), 1008)
        m = FakeModel()
        m.config = types.SimpleNamespace(prompt_encoder_config=types.SimpleNamespace(image_size=1008))
        srv.S["model"] = m
        self.assertEqual(srv.model_image_size(), 1008)

    def test_zoom_crop_refines_small_object_and_rejects_exploding_candidate(self):
        class ZoomModel(FakeModel):
            def __call__(self, **kw):
                self.calls.append(kw)
                if len(self.calls) == 1:
                    # Coarse pass: cand 0 small (score 0.9), cand 1 big (score 0.7)
                    pm = torch.zeros(1, 1, 2, 16, 16)
                    pm[0, 0, 0] = -2.0
                    return types.SimpleNamespace(pred_masks=pm, iou_scores=torch.tensor([[[0.9, 0.7]]]))
                # Zoomed crop pass: cand 0 explodes to whole crop (score 0.99), cand 1 crisp 16x16 (score 0.95)
                pm = torch.zeros(1, 1, 2, 16, 16)
                pm[0, 0, 1] = 5.0
                return types.SimpleNamespace(pred_masks=pm, iou_scores=torch.tensor([[[0.99, 0.95]]]))

        class ZoomProc(FakeProc):
            def post_process_masks(self, masks, sizes, binarize=True):
                h, w = sizes[0].tolist()
                out = torch.full((1, 2, h, w), -5.0)
                if h == 512:
                    # Coarse cand 0: sloppy 20x20 around (256, 256); cand 1: 100x100 enclosing object
                    out[0, 0, 246:266, 246:266] = 5.0
                    out[0, 1, 206:306, 206:306] = 5.0
                else:
                    # Zoomed 256x256 crop centred at (256, 256) -> crop covers [128..384, 128..384]
                    out[0, 0, :, :] = 5.0                  # explodes to crop boundary (must be rejected!)
                    out[0, 1, 120:136, 120:136] = 5.0      # crisp 16x16 centred at (128, 128) -> (256, 256)
                return [out > 0 if binarize else out]

        model = ZoomModel()
        srv.S.update(model=model, proc=ZoomProc(), device="cpu", notice=None)
        res = srv.predict({"image_id": "z1", "size": [512, 512], "points": [[256.0, 256.0]], "labels": [1],
                           "outline": "pixel", "min_area_px": 4}, opaque(512))
        self.assertEqual(len(model.calls), 2)                           # coarse + zoomed crop
        self.assertEqual(len(srv._crop_emb), 1)
        self.assertEqual(len(res["candidates"]), 2)                     # cand 1 preserved for C cycling
        self.assertEqual(res["candidates"][0]["score"], 0.95)           # crisp zoomed cand won over 0.99 exploded
        self.assertEqual(srv._shoelace(res["candidates"][0]["polygons"][0][0]), 256.0)      # 16 x 16
        self.assertEqual(srv._shoelace(res["candidates"][1]["polygons"][0][0]), 10000.0)    # 100 x 100 preserved
        # Cached low-res mask for cand 0 received zoomed logits (5.0 inside crop region)
        lr = srv._masks[res["candidates"][0]["mask_id"]][1]
        self.assertGreater(float(lr.max()), 4.0)

        # Hover skips zoom refinement (only 1 model call)
        model.calls.clear()
        srv.predict({"image_id": "z1", "points": [[256.0, 256.0]], "labels": [1], "hover": True})
        self.assertEqual(len(model.calls), 1)

        # use_cpu clears _crop_emb as well as _emb, while preserving CPU _masks
        srv.S["device"] = "meta"
        srv.use_cpu(RuntimeError("boom"))
        self.assertEqual((len(srv._emb), len(srv._crop_emb)), (0, 0))
        self.assertGreater(len(srv._masks), 0)

    def test_progressive_zoom_two_stage_and_negative_click_ranking(self):
        class ProgModel(FakeModel):
            def __call__(self, **kw):
                self.calls.append(kw)
                pm = torch.zeros(1, 1, 2, 16, 16)
                if len(self.calls) == 1:
                    return types.SimpleNamespace(pred_masks=pm, iou_scores=torch.tensor([[[0.9, 0.7]]]))
                pm[0, 0, 0] = 5.0
                return types.SimpleNamespace(pred_masks=pm, iou_scores=torch.tensor([[[0.95, 0.8]]]))

        class ProgProc(FakeProc):
            def post_process_masks(self, masks, sizes, binarize=True):
                h, w = sizes[0].tolist()
                out = torch.full((1, 2, h, w), -5.0)
                if h == 512:
                    out[0, 0, 246:266, 246:266] = 5.0
                    out[0, 1, 206:306, 206:306] = 5.0
                else:
                    out[0, 0, 120:136, 120:136] = 5.0
                    out[0, 1, 110:146, 110:146] = 5.0
                return [out > 0 if binarize else out]

        model = ProgModel()
        srv.S.update(model=model, proc=ProgProc(), device="cpu", notice=None)
        # Stage 1 (progressive=True): returns coarse immediately with needs_zoom=True
        r1 = srv.predict({"image_id": "p1", "size": [512, 512], "points": [[256.0, 256.0]], "labels": [1],
                          "outline": "pixel", "min_area_px": 4, "progressive": True}, opaque(512))
        self.assertTrue(r1.get("needs_zoom"))
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(srv._shoelace(r1["candidates"][0]["polygons"][0][0]), 400.0)
        # Stage 2 (refine_zoom=True): skips coarse decode via _last_coarse and runs zoom crop only
        r2 = srv.predict({"image_id": "p1", "points": [[256.0, 256.0]], "labels": [1],
                          "outline": "pixel", "min_area_px": 4, "refine_zoom": True})
        self.assertFalse(r2.get("needs_zoom"))
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(srv._shoelace(r2["candidates"][0]["polygons"][0][0]), 256.0)

        # Negative-click ranking prefers candidate that respects exclude click over higher-score violator
        class NegModel(FakeModel):
            def __call__(self, **kw):
                self.calls.append(kw)
                return types.SimpleNamespace(pred_masks=torch.zeros(1, 1, 2, 4, 4),
                                             iou_scores=torch.tensor([[[0.95, 0.80]]]))

        class NegProc(FakeProc):
            def post_process_masks(self, masks, sizes):
                h, w = sizes[0].tolist()
                m = torch.zeros(1, 2, h, w)
                m[0, 0, 2:30, 2:30] = 1                    # covers pos (5,5) AND violates neg (25,25)
                m[0, 1, 2:15, 2:15] = 1                    # covers pos (5,5) and avoids neg (25,25)
                return [m]

        srv.S.update(model=NegModel(), proc=NegProc(), device="cpu", notice=None)
        rn = srv.predict({"image_id": "neg", "size": [64, 64], "points": [[5.5, 5.5], [25.5, 25.5]],
                          "labels": [1, 0], "outline": "pixel", "min_area_px": 4}, opaque(64))
        self.assertEqual(rn["candidates"][0]["score"], 0.8)


class PrewarmAndValidation(Http):
    def test_negative_content_length_400_and_encode_skips_cpu(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.putrequest("POST", "/predict")
        c.putheader("X-Segmentot", "1")
        c.putheader("Content-Length", "-1")
        c.endheaders()
        r = c.getresponse()
        r.read()
        c.close()
        self.assertEqual(r.status, 400)

        srv.S.update(model=FakeModel(), proc=FakeProc(), device="cpu", notice=None)
        hdr = {"X-Segmentot": "1", "Content-Type": "application/octet-stream",
               "X-Segmentot-Request": json.dumps({"image_id": "enc1", "size": [16, 16]}, separators=(",", ":"))}
        code, res = self.ask("POST", "/encode", hdr, opaque())
        self.assertEqual(code, 200)
        self.assertEqual(res.get("encoded"), False)
        self.assertEqual(res.get("skipped"), "cpu")


if __name__ == "__main__":
    unittest.main()
