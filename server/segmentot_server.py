"""Segmentot click-segmentation engine (SAM 2.1, EdgeTAM or SAM 3 via Hugging Face transformers).

Runs in Segmentot's own Python environment (created by the plugin), never inside QGIS.
Listens on 127.0.0.1 only. Exits when idle for --idle-exit seconds or when the
QGIS process that started it (--parent-pid) is gone. Uses the GPU (CUDA, then Apple MPS)
while it works; if it fails the engine moves to the CPU for good and says so in "notice".

Every request needs an X-Segmentot header and Host 127.0.0.1:<port> or localhost:<port>,
else 403 (keeps web pages in the user's browser out). Bodies over 64 MB get 413.

  GET  /health   -> {"ok", "engine", "device", "gpu", "model", "loaded", "torch", "notice"?}
                    X-Segmentot: load loads the model before answering.
  POST /predict  {image_id, image?, points:[[x,y]..], labels:[1|0..], outline, min_area_px}
                 image: base64 PNG; pixels with alpha 0 are no-data, a PNG without alpha is all valid.
                 -> {"candidates": [{"polygons": [[ext, hole..], ..], "score": s}, ..],
                     "device", "seconds", "notice"?}
                    best first, masks covering the clicks before those that miss them.
                    Polygons are closed rings in pixel-corner coordinates (x right, y down).
                    {"need_image": true, ..} when image_id is unknown and no image was sent.
                    {"candidates": [], "error": "No imagery here", ..} when under 1% is valid.
  "device" is the device in use now; "notice" is only there after the GPU failed.
"""
import argparse
import base64
import io
import json
import os
import sys
import threading
import time
import traceback
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2
import numpy as np
import torch
from PIL import Image
from rasterio.features import shapes

CACHE_IMAGES = 6
MAX_BODY = 64 * 2**20           # bytes
MAX_PIXELS = 4096 * 4096        # the client sends 1024 x 1024
_lock = threading.Lock()
_last_request = time.time()
_emb = OrderedDict()            # image_id -> (embeddings, (h, w), valid mask)
S = {"proc": None, "model": None, "device": "cpu", "model_id": None, "notice": None}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[segmentot]", *a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- model
def pick_device(want):
    if want != "auto":
        return want
    for dev, ok in (("cuda", torch.cuda.is_available()),
                    ("mps", getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available())):
        if ok:
            try:          # a real kernel: catches GPUs this PyTorch build has no code for
                float(torch.nn.functional.conv2d(torch.ones(1, 3, 8, 8, device=dev),
                                                 torch.ones(4, 3, 3, 3, device=dev)).sum())
                return dev
            except Exception as e:
                log(f"{dev} not usable ({str(e).splitlines()[0][:150]}), using the CPU")
    return "cpu"


# part of the model id -> (model class, processor class). EdgeTAM and SAM 3 are called exactly like SAM 2.
# (SAM-HQ was tried too: with transformers 5.23 it returns misplaced masks, so it is not offered.)
FAMILIES = (("edgetam", "EdgeTamModel", "Sam2Processor"),
            ("sam3", "Sam3TrackerModel", "Sam3TrackerProcessor"))


def family(model_id):
    for part, model, proc in FAMILIES:
        if part in model_id.lower():
            return model, proc
    return "Sam2Model", "Sam2Processor"


def load():
    if S["model"] is None:
        import transformers
        transformers.utils.logging.set_verbosity_error()
        t = time.time()
        model, proc = family(S["model_id"])
        S["proc"] = getattr(transformers, proc).from_pretrained(S["model_id"])
        S["model"] = getattr(transformers, model).from_pretrained(S["model_id"]).eval()
        try:
            S["model"].to(S["device"])
        except Exception as e:
            if S["device"] == "cpu":
                raise
            use_cpu(e)
        log(f"{S['model_id']} loaded on {S['device']} in {time.time() - t:.1f}s")
    return S["proc"], S["model"]


def use_cpu(err):
    """The GPU failed: tell the user, free the GPU and run on the CPU from now on."""
    first = (str(err).strip().splitlines() or [type(err).__name__])[0][:150]
    S["notice"] = f"GPU failed ({first}); now running on the CPU (slower)."
    log(S["notice"] + "\n" + traceback.format_exc())
    gpu, S["device"] = S["device"], "cpu"
    _emb.clear()                                  # cached embeddings live on the GPU
    if S["model"] is not None:
        S["model"].to("cpu")
    traceback.clear_frames(err.__traceback__)     # drops the failed call's GPU tensors so they can be freed
    if gpu == "cuda":
        torch.cuda.empty_cache()
    elif gpu == "mps":
        torch.mps.empty_cache()


def gpu_name():
    if S["device"] == "cuda":
        return torch.cuda.get_device_name(0)
    if S["device"] == "mps":
        return "Apple GPU (MPS)"
    return None


# ---------------------------------------------------------------- masks -> polygons
def decode_image(b64):
    """PNG -> (RGB array, valid mask). Pixels with alpha 0 are no-data; without alpha all are valid."""
    im = Image.open(io.BytesIO(base64.b64decode(b64)))
    if im.width * im.height > MAX_PIXELS:
        raise ValueError(f"image too large ({im.width}x{im.height} px)")
    a = np.asarray(im.convert("RGBA"))
    return a[..., :3], a[..., 3] > 0


def _shoelace(ring):
    a = np.asarray(ring)
    return 0.5 * abs(np.dot(a[:-1, 0], a[1:, 1]) - np.dot(a[1:, 0], a[:-1, 1]))


def _ring(r, outline):
    """A closed ring on pixel edges -> [[x, y], ..]; "smooth" simplifies it to within 1 px."""
    if outline != "pixel":
        s = cv2.approxPolyDP(np.float32(r).reshape(-1, 1, 2), 1.0, True).reshape(-1, 2)
        if len(s) >= 3:                # a 1 px wide strip simplifies to a line: keep it exact
            r = list(s) + [s[0]]
    return [[float(x), float(y)] for x, y in r]


def mask_to_polygons(mask, min_area_px, outline):
    """Binary mask -> [[exterior, hole, ...], ...] in pixel-corner coordinates."""
    m = mask.astype(np.uint8)
    polys = []
    # pixel-edge rings for both outlines: cv2.findContours traces pixel centres and shrinks by 0.5 px
    for geom, _ in shapes(m, mask=m.astype(bool), connectivity=4):
        ext, *holes = geom["coordinates"]
        if _shoelace(ext) >= min_area_px:
            polys.append([_ring(r, outline) for r in [ext] + [h for h in holes if _shoelace(h) >= min_area_px]])
    return polys


def keep_clicked(polys, pts, lbl):
    """The part(s) containing a positive click, else the part nearest to one."""
    pos = [p for p, l in zip(pts, lbl) if l == 1]

    def dist(rings):
        ext = np.array(rings[0], dtype=np.float32)
        return min(-cv2.pointPolygonTest(ext, (float(x), float(y)), True) for x, y in pos)
    hits = [p for p in polys if dist(p) <= 0]
    return hits or sorted(polys, key=dist)[:1]


# ---------------------------------------------------------------- job
def predict(req):
    """run_predict; if the GPU fails (out of memory, an op MPS lacks, ..) move to the CPU and retry once."""
    try:
        return run_predict(req)
    except RuntimeError as e:                     # torch's OutOfMemoryError and NotImplementedError too
        if S["device"] == "cpu" or S["model"] is None:      # load() handles its own GPU failure
            raise
        use_cpu(e)
    return run_predict(req)


def run_predict(req):
    pts = np.array(req["points"], dtype=np.float32)
    lbl = np.array(req["labels"], dtype=np.int64)
    # checked here so a bad request cannot fail inside torch and pass for a GPU failure
    if not len(pts) or pts.shape[1:] != (2,) or lbl.shape != pts.shape[:1] or not np.isin(lbl, (0, 1)).all():
        raise ValueError("points must be [[x, y], ..] with one label (1 or 0) each")
    proc, model = load()
    dev = S["device"]
    image_id = req.get("image_id")
    with torch.inference_mode():
        if image_id in _emb:
            emb, (h, w), valid = _emb[image_id]
            _emb.move_to_end(image_id)
        else:
            if not req.get("image"):
                return {"need_image": True}
            img, valid = decode_image(req["image"])
            h, w = img.shape[:2]
            if valid.mean() < 0.01:
                return {"candidates": [], "error": "No imagery here"}
            pv = proc(images=img, return_tensors="pt")["pixel_values"].to(dev)
            emb = model.get_image_embeddings(pv)
            _emb[image_id] = (emb, (h, w), valid)
            while len(_emb) > CACHE_IMAGES:
                _emb.popitem(last=False)
        size = float(model.config.prompt_encoder_config.image_size)      # 1024; 1008 for SAM 3
        scaled = pts * np.array([size / w, size / h], dtype=np.float32)
        multi = len(pts) == 1
        out = model(image_embeddings=emb,
                    input_points=torch.from_numpy(scaled)[None, None].to(dev),
                    input_labels=torch.from_numpy(lbl)[None, None].to(dev),
                    multimask_output=multi)
        masks = proc.post_process_masks(out.pred_masks.float().cpu(), torch.tensor([[h, w]]))[0][0]
        scores = out.iou_scores[0, 0].float().cpu().numpy()
    masks = masks.numpy().astype(bool)
    min_area = float(req.get("min_area_px", 4))
    outline = req.get("outline", "pixel")
    pos = [(min(h - 1, max(0, int(y))), min(w - 1, max(0, int(x)))) for (x, y), l in zip(pts, lbl) if l == 1]
    out = []
    for i in np.argsort(-scores):
        m = masks[i] & valid
        polys = mask_to_polygons(m, min_area, outline)
        if polys:
            covers = sum(bool(m[r, c]) for r, c in pos)
            out.append((covers, float(scores[i]), keep_clicked(polys, pts, lbl)))
    out.sort(key=lambda o: (-o[0], -o[1]))
    return {"candidates": [{"polygons": p, "score": round(sc, 4)} for _, sc, p in out]}


# ---------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _refused(self):
        # A web page can send a simple POST here but cannot add this header without a CORS
        # preflight we never answer; DNS rebinding shows up as a foreign Host.
        port = self.server.server_port
        if self.headers.get("X-Segmentot") and self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}"):
            return False
        self._send(403, {"error": "forbidden"})
        return True

    def do_GET(self):
        global _last_request
        if self._refused():
            return
        _last_request = time.time()
        if self.path != "/health":
            return self._send(404, {"error": "not found"})
        try:
            if "load" in self.headers.get("X-Segmentot", ""):
                with _lock:
                    load()
            res = {"ok": True, "engine": "segmentot", "device": S["device"], "gpu": gpu_name(),
                   "loaded": S["model"] is not None, "model": S["model_id"].split("/")[-1],
                   "torch": torch.__version__}
            if S["notice"]:
                res["notice"] = S["notice"]
            self._send(200, res)
        except Exception as e:
            log(traceback.format_exc())
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        global _last_request
        if self._refused():
            return
        _last_request = time.time()
        if self.path != "/predict":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > MAX_BODY:
                return self._send(413, {"error": "request too large"})
            req = json.loads(self.rfile.read(n))
            t = time.time()
            with _lock:
                res = predict(req)
                res["device"] = S["device"]
                if S["notice"]:
                    res["notice"] = S["notice"]
            res["seconds"] = round(time.time() - t, 3)
            self._send(200, res)
        except Exception as e:
            log(traceback.format_exc())
            self._send(500, {"error": f"{type(e).__name__}: {e}"})
        finally:
            _last_request = time.time()


class Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = os.name != "nt"         # on Windows it lets a second engine bind the same port


def pid_alive(pid):
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return code.value == 259                         # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def watchdog(idle_exit, parent_pid):
    while True:
        time.sleep(5)
        if parent_pid and not pid_alive(parent_pid):
            log("QGIS is gone, exiting")
            os._exit(0)
        if idle_exit and not _lock.locked() and time.time() - _last_request > idle_exit:
            log("idle, exiting")
            os._exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8776)
    ap.add_argument("--model", default="facebook/sam2.1-hiera-base-plus")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--idle-exit", type=int, default=900)
    ap.add_argument("--parent-pid", type=int, default=0)
    a = ap.parse_args()
    S["model_id"], S["device"] = a.model, pick_device(a.device)
    srv = Server(("127.0.0.1", a.port), Handler)
    threading.Thread(target=watchdog, args=(a.idle_exit, a.parent_pid), daemon=True).start()
    log(f"listening on 127.0.0.1:{a.port}, device={S['device']}, model={a.model}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
