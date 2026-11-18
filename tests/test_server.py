"""Tests for server/segmentot_server.py. Needs torch, numpy, pillow, opencv and rasterio; loads no model.

  uv run --no-project --with torch --with numpy --with pillow --with opencv-python-headless \
      --with rasterio python -m unittest tests/test_server.py -v
"""
import base64
import http.client
import importlib.util
import io
import json
import os
import sys
import threading
import types
import unittest
from unittest import mock

import numpy as np
import torch
from PIL import Image

_spec = importlib.util.spec_from_file_location(
    "segmentot_server", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server", "segmentot_server.py"))
srv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(srv)


def png_b64(arr, mode):
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


class Clean(unittest.TestCase):
    def setUp(self):
        self._s = dict(srv.S)
        srv.S["model_id"] = "facebook/sam2.1-hiera-tiny"
        srv._emb.clear()

    def tearDown(self):
        srv.S.clear()
        srv.S.update(self._s)
        srv._emb.clear()


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


class Alpha(unittest.TestCase):
    def test_rgba_black_opaque_valid_transparent_invalid(self):
        a = np.zeros((2, 3, 4), np.uint8)          # all black
        a[0, :, 3] = 255                           # row 0 opaque
        a[1, 0, 3] = 1                             # barely opaque still counts
        rgb, valid = srv.decode_image(png_b64(a, "RGBA"))
        self.assertEqual(rgb.shape, (2, 3, 3))
        self.assertEqual(valid.tolist(), [[True, True, True], [True, False, False]])

    def test_rgb_all_valid(self):
        rgb, valid = srv.decode_image(png_b64(np.zeros((4, 5, 3), np.uint8), "RGB"))
        self.assertEqual(rgb.shape, (4, 5, 3))
        self.assertTrue(valid.all())

    def test_huge_image_rejected(self):
        buf = io.BytesIO()
        Image.new("1", (5000, 4000)).save(buf, "PNG")
        with self.assertRaises(ValueError):
            srv.decode_image(base64.b64encode(buf.getvalue()).decode())


# --------------------------------------------------------------- fakes for the GPU -> CPU fallback
def _gpu_check():
    if srv.S["device"] != "cpu":
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB.\nmore detail")


class FakeModel:
    config = types.SimpleNamespace(prompt_encoder_config=types.SimpleNamespace(image_size=1024))

    def __init__(self):
        self.moved = []

    def to(self, dev):
        self.moved.append(dev)
        return self

    def get_image_embeddings(self, pv):
        _gpu_check()
        return torch.zeros(1)

    def __call__(self, **kw):
        _gpu_check()
        return types.SimpleNamespace(pred_masks=torch.zeros(1, 1, 1, 4, 4), iou_scores=torch.tensor([[[0.9]]]))


class FakeProc:
    def __call__(self, images, return_tensors):
        return {"pixel_values": torch.zeros(1, 3, 4, 4)}

    def post_process_masks(self, masks, sizes):
        h, w = sizes[0].tolist()
        m = torch.zeros(1, 1, h, w)
        m[0, 0, 2:12, 2:12] = 1                    # a 10 x 10 px object
        return [m]


def opaque_png(n=16):
    return png_b64(np.full((n, n, 4), 255, np.uint8), "RGBA")


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
        res = srv.predict({"image_id": "b", "image": opaque_png(), "points": [[5, 5]], "labels": [1],
                           "outline": "pixel", "min_area_px": 4})
        self.assertEqual(srv.S["device"], "cpu")
        self.assertEqual([[srv._shoelace(r) for r in rings] for c in res["candidates"] for rings in c["polygons"]],
                         [[100]])
        self.assertIn("b", srv._emb)

    def test_no_fallback_for_bad_request(self):
        with self.assertRaises(ValueError):
            srv.predict({"image_id": "c", "image": opaque_png(), "points": [[5, 5]], "labels": [1, 0]})
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
            srv.predict({"image_id": "d", "image": opaque_png(), "points": [[5, 5]], "labels": [1]})
        self.assertIsNone(srv.S["notice"])

    def test_each_model_family_has_its_loader(self):
        self.assertEqual(srv.family("facebook/sam2.1-hiera-tiny"), ("Sam2Model", "Sam2Processor"))
        self.assertEqual(srv.family("yonigozlan/EdgeTAM-hf"), ("EdgeTamModel", "Sam2Processor"))
        self.assertEqual(srv.family("facebook/sam3"), ("Sam3TrackerModel", "Sam3TrackerProcessor"))

    def test_load_falls_back_when_move_to_gpu_fails(self):
        model = FakeModel()
        model.eval = lambda: model
        model.to = mock.Mock(side_effect=[RuntimeError("CUDA error: no kernel image"), model])
        tf = types.SimpleNamespace(
            Sam2Model=types.SimpleNamespace(from_pretrained=lambda _: model),
            Sam2Processor=types.SimpleNamespace(from_pretrained=lambda _: FakeProc()),
            utils=types.SimpleNamespace(logging=types.SimpleNamespace(set_verbosity_error=lambda: None)))
        srv.S.update(model=None, proc=None, device="cuda")
        with mock.patch.dict(sys.modules, {"transformers": tf}), \
                mock.patch.object(srv.torch.cuda, "empty_cache"):
            srv.load()
        self.assertIs(srv.S["model"], model)
        self.assertEqual([c.args for c in model.to.call_args_list], [("cuda",), ("cpu",)])
        self.assertEqual(srv.S["device"], "cpu")
        self.assertTrue(srv.S["notice"].startswith("GPU failed (CUDA error: no kernel image)"))


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
            self.assertNotIn("notice", r)

    def test_predict_reports_fallback(self):
        srv.S.update(model=FakeModel(), proc=FakeProc(), device="meta", notice=None)
        h = {"X-Segmentot": "1", "Content-Type": "application/json"}
        body = json.dumps({"image_id": "e", "image": opaque_png(), "points": [[5, 5]], "labels": [1]})
        code, r = self.ask("POST", "/predict", h, body)
        self.assertEqual(code, 200)
        self.assertEqual(len(r["candidates"]), 1)
        self.assertEqual(r["device"], "cpu")
        self.assertTrue(r["notice"].startswith("GPU failed ("))
        self.assertIn("seconds", r)
        code, r = self.ask("GET", "/health", {"X-Segmentot": "1"})
        self.assertEqual((code, r["device"], r["notice"][:11]), (200, "cpu", "GPU failed "))


if __name__ == "__main__":
    unittest.main()
