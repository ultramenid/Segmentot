"""Segmentot click-segmentation engine (SAM 2.1, EdgeTAM or SAM 3 via Hugging Face transformers).

Runs in Segmentot's own Python environment (created by the plugin), never inside QGIS.
Listens on 127.0.0.1 only. Exits when idle for --idle-exit seconds or when the
QGIS process that started it (--parent-pid) is gone. Uses the GPU (CUDA, then Apple MPS)
while it works; if it fails the engine moves to the CPU for good and says so in "notice".

Every request needs an X-Segmentot header and Host 127.0.0.1:<port> or localhost:<port>,
else 403 (keeps web pages in the user's browser out). Bodies over 64 MB get 413.
A request with an X-Segmentot-Nonce header gets X-Segmentot-Proof back: HMAC-SHA256 of "<port>:<nonce>"
with the key in SEGMENTOT_KEY. The plugin sends imagery only to an engine that proves itself this way.

  GET  /health   -> {"ok", "engine", "api", "device", "gpu", "model", "image_size", "loaded", "torch", "notice"?}
                    X-Segmentot: load loads the model before answering.
  POST /predict  {image_id, points:[[x,y]..], labels:[1|0..], outline, min_area_px, mask_id?, hover?}
                 With the image: this JSON, plus "size": [w, h], goes in the X-Segmentot-Request header and the
                 body (application/octet-stream) is the pixels, RGBA8888 row by row, w * h * 4 bytes;
                 alpha 0 is no-data.
                 -> {"candidates": [{"polygons": [[ext, hole..], ..], "score": s, "mask_id": id}, ..],
                     "image_size", "device", "seconds", "notice"?}
                    best first, masks covering the clicks before those that miss them.
                    Polygons are closed rings in pixel-corner coordinates (x right, y down).
                    {"need_image": true, ..} when image_id is unknown and no image was sent.
                    {"candidates": [], "error": "No imagery here", ..} when no valid imagery is near the clicks.
  "device" is the device in use now; "notice" is only there after the GPU failed.
"""
import argparse
import hmac
import json
import math
import os
import sys
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from affine import Affine
from rasterio.features import shapes

API = 2                         # request format; the plugin checks it in /health (core/engine.py has the same)
CACHE_IMAGES = 24               # warm image embeddings (hybrid: top CACHE_HOT on GPU/MPS, rest FP16 on CPU)
CACHE_HOT = 2
CACHE_MASKS = 32
CACHE_CROPS = 12
CACHE_CROP_HOT = 2
MAX_BODY = 64 * 2**20           # bytes
MAX_PIXELS = 4096 * 4096        # the client sends 1024 x 1024
KEY = os.environ.get("SEGMENTOT_KEY", "").encode()      # set by the plugin that starts this engine
_lock = threading.Lock()
_gen_lock = threading.Lock()
_hover_gen = 0
_pending_clicks = 0
_last_request = time.time()
_emb = OrderedDict()            # image_id -> (embeddings, (h, w), valid mask, rgb img)
_crop_emb = OrderedDict()       # (image_id, cx0, cy0, cx1, cy1) -> crop embeddings
_masks = OrderedDict()          # mask_id -> (image_id, low-res logits cpu float32 (1, 1, H_lr, W_lr))
_norm_cache = {}                # (id(proc), dev) -> (mean_t, std_t, rescale, do_norm, do_rescale)
_last_coarse = [None]           # 1-slot Stage-1 coarse cache for two-stage progressive zoom refinement
S = {"proc": None, "model": None, "device": "cpu", "model_id": None, "notice": None}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[segmentot]", *a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- model & cache helpers
def _tune_backend(dev):
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
        except (RuntimeError, ValueError) as e:
            log(f"could not set matmul precision: {e}")
    if dev == "cpu":
        try:
            torch.set_num_threads(min(os.cpu_count() or 4, 16))
        except (RuntimeError, ValueError) as e:
            log(f"could not tune CPU threads: {e}")
    elif dev == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _preferred_gpu_dtype(dev):
    if dev == "cuda" and torch.cuda.is_available():
        try:
            major, _ = torch.cuda.get_device_capability(0)
            if major >= 8:              # Ampere/Ada/Hopper/Blackwell: native bf16, no fp16 overflow
                return torch.bfloat16
            if major < 7:               # Pascal/Maxwell: fp16 is slow or unsupported
                return torch.float32
        except RuntimeError as e:
            log(f"could not query CUDA compute capability: {e}")
    return torch.float16


def _move_emb(x, dev, dt):
    if isinstance(x, torch.Tensor):
        want_dt = dt if (dt is not None and x.is_floating_point()) else None
        if x.device.type == dev and (want_dt is None or x.dtype == want_dt):
            return x
        return x.to(dev, want_dt, non_blocking=(dev == "cuda")) if want_dt is not None else x.to(
            dev, non_blocking=(dev == "cuda"))
    if isinstance(x, list):
        return [_move_emb(v, dev, dt) for v in x]
    if isinstance(x, tuple):
        return tuple(_move_emb(v, dev, dt) for v in x)
    if isinstance(x, dict):
        return {k: _move_emb(v, dev, dt) for k, v in x.items()}
    return x


def _trim_hot_emb(dev):
    if dev not in ("cuda", "mps") or len(_emb) <= CACHE_HOT:
        return
    keys = list(_emb.keys())[:-CACHE_HOT]
    for k in keys:
        entry = _emb[k]
        emb = _move_emb(entry[0], "cpu", None)
        _emb[k] = (emb,) + entry[1:]


def _trim_hot_crops(dev):
    if dev not in ("cuda", "mps") or len(_crop_emb) <= CACHE_CROP_HOT:
        return
    keys = list(_crop_emb.keys())[:-CACHE_CROP_HOT]
    for k in keys:
        _crop_emb[k] = _move_emb(_crop_emb[k], "cpu", None)


def _is_hf_proc(proc):
    ip = getattr(proc, "image_processor", None)
    if ip is None:
        return False
    mean = getattr(ip, "image_mean", None)
    std = getattr(ip, "image_std", None)
    return (isinstance(mean, (list, tuple)) and len(mean) == 3
            and isinstance(std, (list, tuple)) and len(std) == 3)


def _nodata_fill(proc):
    if _is_hf_proc(proc):
        mean = np.asarray(proc.image_processor.image_mean, dtype=np.float32)
        return np.clip(np.rint(mean * 255.0), 0, 255).astype(np.uint8)
    return np.array([124, 116, 104], dtype=np.uint8)


def _norm_params(proc, dev):
    key = (id(proc), dev)
    p = _norm_cache.get(key)
    if p is not None:
        return p
    ip = proc.image_processor
    mean_t = torch.tensor([float(v) for v in ip.image_mean], dtype=torch.float32, device=dev).view(1, 3, 1, 1)
    std_t = torch.tensor([float(v) for v in ip.image_std], dtype=torch.float32, device=dev).view(1, 3, 1, 1)
    rescale = float(getattr(ip, "rescale_factor", 1.0 / 255.0))
    do_norm = bool(getattr(ip, "do_normalize", True))
    do_rescale = bool(getattr(ip, "do_rescale", True))
    p = (mean_t, std_t, rescale, do_norm, do_rescale)
    _norm_cache[key] = p
    return p


def _preprocess(proc, img, size, dev, dt):
    if not _is_hf_proc(proc) or dev == "meta" or img.ndim != 3 or img.shape[2] != 3:
        return proc(images=img, return_tensors="pt")["pixel_values"].to(dev, dt)
    mean_t, std_t, rescale, do_norm, do_rescale = _norm_params(proc, dev)
    arr = np.ascontiguousarray(img)
    if not arr.flags.writeable:
        arr = arr.copy()
    t = torch.from_numpy(arr).to(dev, non_blocking=(dev == "cuda"))
    t = t.permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    sz = int(size)
    if t.shape[2] != sz or t.shape[3] != sz:
        aa = t.shape[2] > sz or t.shape[3] > sz
        t = F.interpolate(t, size=(sz, sz), mode="bilinear", align_corners=False, antialias=aa)
    if do_rescale:
        t.mul_(rescale)
    if do_norm:
        t.sub_(mean_t).div_(std_t)
    return t.to(dt)


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
        _tune_backend(S["device"])
        model, proc = family(S["model_id"])
        S["proc"] = getattr(transformers, proc).from_pretrained(S["model_id"])
        S["model"] = getattr(transformers, model).from_pretrained(S["model_id"]).eval()
        if S["device"] != "cpu":                  # the CPU stays in float32 and has nothing to warm up
            pref = _preferred_gpu_dtype(S["device"])
            try:
                if pref == torch.float32:
                    S["model"].to(S["device"], torch.float32)
                    if not warm_up():
                        raise RuntimeError("GPU warm-up returned non-finite outputs")
                else:
                    try:                          # float16 / bfloat16 halves the GPU time
                        S["model"].to(S["device"], pref)
                        ok = warm_up()
                    except Exception:             # some GPUs fail in half precision instead of giving NaN
                        ok = False
                    if not ok:
                        log(f"{pref} does not work on this GPU, using float32")
                        S["model"].to(S["device"], torch.float32)
                        if not warm_up():
                            raise RuntimeError("GPU warm-up returned non-finite outputs")
            except Exception as e:
                use_cpu(e)
        log(f"{S['model_id']} loaded on {S['device']} ({S['model'].dtype}) in {time.time() - t:.1f}s")
    return S["proc"], S["model"]


def warm_up():
    """One dummy image and click on the GPU: its one-time setup lands in loading, not in the first click.
    False if the outputs are not finite (float16 on some GPUs, e.g. GTX 16xx)."""
    proc, model, dev = S["proc"], S["model"], S["device"]
    sz = model_image_size()
    noise = np.random.default_rng(0).integers(0, 256, (sz, sz, 3), np.uint8)    # zeros may not overflow
    with torch.inference_mode():
        pv = _preprocess(proc, noise, sz, dev, model.dtype)
        out = model(image_embeddings=model.get_image_embeddings(pv),
                    input_points=torch.full((1, 1, 1, 2), (sz - 1) / 2.0, dtype=torch.float32, device=dev),
                    input_labels=torch.ones((1, 1, 1), dtype=torch.int64, device=dev), multimask_output=True)
        return bool(out.pred_masks.isfinite().all() and out.iou_scores.isfinite().all())


def use_cpu(err):
    """The GPU failed: tell the user, free the GPU and run on the CPU from now on."""
    first = (str(err).strip().splitlines() or [type(err).__name__])[0][:150]
    S["notice"] = f"GPU failed ({first}); now running on the CPU (slower)."
    log(S["notice"] + "\n" + traceback.format_exc())
    gpu, S["device"] = S["device"], "cpu"
    _emb.clear()                                  # cached embeddings live on the GPU
    _crop_emb.clear()
    _norm_cache.clear()
    _last_coarse[0] = None
    _tune_backend("cpu")
    if S["model"] is not None:
        S["model"].to("cpu", torch.float32)
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


def model_image_size():
    if S["model"] is not None:
        return int(S["model"].config.prompt_encoder_config.image_size)
    return 1008 if "sam3" in (S["model_id"] or "").lower() else 1024


# ---------------------------------------------------------------- masks -> polygons
def _shoelace(ring):
    a = np.asarray(ring)
    return 0.5 * abs(np.dot(a[:-1, 0], a[1:, 1]) - np.dot(a[1:, 0], a[:-1, 1]))


def _ring(r, outline, eps=0.7):
    """A closed ring on pixel edges -> [[x, y], ..]; "smooth" simplifies it."""
    if outline != "pixel":
        s = cv2.approxPolyDP(np.float32(r[:-1]).reshape(-1, 1, 2), eps, True).reshape(-1, 2)
        if len(s) >= 3:                # a 1 px wide strip simplifies to a line: keep it exact
            r = list(s) + [s[0]]
    return [[float(x), float(y)] for x, y in r]


def _tight_bounds(mask, pad=0):
    rows = np.flatnonzero(mask.any(axis=1))
    if not len(rows):
        return None
    h, w = mask.shape
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    cols = np.flatnonzero(mask[y0:y1].any(axis=0))
    if not len(cols):
        return None
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    if pad:
        y0, y1 = max(0, y0 - pad), min(h, y1 + pad)
        x0, x1 = max(0, x0 - pad), min(w, x1 + pad)
    return x0, y0, x1, y1


def mask_to_polygons(mask, min_area_px, outline):
    """Binary mask -> [[exterior, hole, ...], ...] in pixel-corner coordinates."""
    bounds = _tight_bounds(mask)
    if bounds is None:
        return []
    x0, y0, x1, y1 = bounds
    sub = np.ascontiguousarray(mask[y0:y1, x0:x1], dtype=np.uint8)
    if int(sub.sum()) < min_area_px:
        return []
    polys = []
    # pixel-edge rings for both outlines: cv2.findContours traces pixel centres and shrinks by 0.5 px
    for geom, _ in shapes(sub, mask=sub.astype(bool), connectivity=4, transform=Affine.translation(x0, y0)):
        ext, *holes = geom["coordinates"]
        if _shoelace(ext) >= min_area_px:
            polys.append([_ring(r, outline) for r in [ext] + [h for h in holes if _shoelace(h) >= min_area_px]])
    return polys


def logits_to_polygons(logits, valid, min_area_px):
    """Continuous float logits -> sub-pixel smooth polygons in pixel-corner coordinates."""
    m = (logits > 0) & valid
    bounds = _tight_bounds(m, pad=2)
    if bounds is None:
        return []
    x0, y0, x1, y1 = bounds
    bh, bw = y1 - y0, x1 - x0
    k = 4 if bh * bw <= 16384 else 2
    crop = np.ascontiguousarray(logits[y0:y1, x0:x1], dtype=np.float32)
    hi = cv2.resize(crop, (bw * k, bh * k), interpolation=cv2.INTER_LINEAR) > 0
    v_crop = valid[y0:y1, x0:x1]
    if not v_crop.all():
        hi &= np.repeat(np.repeat(v_crop, k, axis=0), k, axis=1)
    min_a = min_area_px * k * k
    if int(hi.sum()) < min_a:
        return []
    offset = np.array([x0, y0], dtype=np.float32)
    polys = []
    for geom, _ in shapes(hi.astype(np.uint8), mask=hi, connectivity=4):
        ext, *holes = geom["coordinates"]
        if _shoelace(ext) >= min_a:
            rings = []
            for r in [ext] + [hl for hl in holes if _shoelace(hl) >= min_a]:
                arr = np.asarray(r[:-1], dtype=np.float32) / k + offset
                s = cv2.approxPolyDP(arr.reshape(-1, 1, 2), 0.5, True).reshape(-1, 2)
                pts = s if len(s) >= 3 else arr
                rings.append([[float(x), float(y)] for x, y in pts] + [[float(pts[0][0]), float(pts[0][1])]])
            polys.append(rings)
    return polys


def _poly_dist(rings, x, y):
    """Signed distance from (x, y) to a polygon with holes: <= 0 iff inside exterior and outside all holes."""
    p = (float(x), float(y))
    d = -cv2.pointPolygonTest(np.array(rings[0], dtype=np.float32), p, True)
    for h in rings[1:]:
        d = max(d, cv2.pointPolygonTest(np.array(h, dtype=np.float32), p, True))
    return d


def keep_clicked(polys, pts, lbl):
    """The part(s) containing a positive click, else the part nearest to one."""
    pos = [(float(x), float(y)) for (x, y), lb in zip(pts, lbl) if lb == 1]
    if not pos:
        return polys[:1]

    def dist(rings):
        return min(_poly_dist(rings, x, y) for x, y in pos)
    hits = [p for p in polys if dist(p) <= 0]
    return hits or sorted(polys, key=dist)[:1]


def _poly_bbox(polys):
    xs = [x for rings in polys for x, _ in rings[0]]
    ys = [y for rings in polys for _, y in rings[0]]
    return min(xs), min(ys), max(xs), max(ys)


def _post_process(proc, pred_masks, h, w, need_logits=True):
    if _is_hf_proc(proc) and pred_masks.ndim == 5:
        pm = pred_masks[0, 0]
        up = F.interpolate(pm.unsqueeze(0).float(), size=(h, w), mode="bilinear", align_corners=False)[0]
        if need_logits:
            lg = up.cpu().numpy()
            return lg > 0, lg
        return (up > 0).cpu().numpy(), None
    pm_cpu = pred_masks.float().cpu()
    sizes = torch.tensor([[h, w]])
    try:
        logits = proc.post_process_masks(pm_cpu, sizes, binarize=False)[0][0].numpy()
        return logits > 0, logits
    except TypeError:
        m = proc.post_process_masks(pm_cpu, sizes)[0][0].numpy().astype(bool)
        return m, None


def _put_mask(image_id, logits):
    mid = uuid.uuid4().hex[:16]
    _masks[mid] = (image_id, logits.detach().float().cpu())
    while len(_masks) > CACHE_MASKS:
        _masks.popitem(last=False)
    return mid


def _zoom_crop_key(image_id, bbox, h, w):
    """Compute (image_id, cx0, cy0, cx1, cy1) for single-click zoom refinement, or None if not applicable."""
    bx0, by0, bx1, by1 = bbox
    bw, bh = bx1 - bx0, by1 - by0
    if max(bw, bh) < 40.0:
        crop_sz = 256
    else:
        crop_sz = min(min(h, w), max(256, int(math.ceil(max(bw, bh) * 3.0 / 32.0)) * 32))
        if crop_sz >= int(0.75 * min(h, w)):
            return None
    if crop_sz <= 0 or crop_sz > min(h, w):
        return None
    cx0 = max(0, min(w - crop_sz, int(round((bx0 + bx1) / 2.0 - crop_sz / 2.0))))
    cy0 = max(0, min(h - crop_sz, int(round((by0 + by1) / 2.0 - crop_sz / 2.0))))
    cx0 = max(0, min(w - crop_sz, (cx0 // 4) * 4))
    cy0 = max(0, min(h - crop_sz, (cy0 // 4) * 4))
    return (image_id, cx0, cy0, cx0 + crop_sz, cy0 + crop_sz)


def _find_cached_crop(image_id, bbox, pts, h, w):
    """Return a cached crop key (image_id, cx0, cy0, cx1, cy1) framing all clicks and bbox, or None."""
    bx0, by0, bx1, by1 = bbox
    for key in reversed(_crop_emb):
        if key[0] != image_id:
            continue
        _, cx0, cy0, cx1, cy1 = key
        if cx0 < 0 or cy0 < 0 or cx1 > w or cy1 > h or cx1 <= cx0 or cy1 <= cy0:
            continue
        if (cx0 > 0 and bx0 < cx0 + 12) or (cx1 < w and bx1 > cx1 - 12):
            continue
        if (cy0 > 0 and by0 < cy0 + 12) or (cy1 < h and by1 > cy1 - 12):
            continue
        if any((cx0 > 0 and x < cx0 + 8) or (cx1 < w and x > cx1 - 8)
               or (cy0 > 0 and y < cy0 + 8) or (cy1 < h and y > cy1 - 8) for x, y in pts):
            continue
        return key
    return None


def _zoom_refine(proc, model, dev, dt, size, image_id, img, valid,
                 pts, lbl, pos_pts, neg_pts, outline, min_area, bbox, orig_lr,
                 orig_covers, orig_neg_viol, coarse_mask=None, crop_key=None):
    """Zoom onto a crop around a small object and refine its mask with bilinear upsampling."""
    h, w = valid.shape
    bx0, by0, bx1, by1 = bbox
    bw, bh = bx1 - bx0, by1 - by0
    if crop_key is not None:
        _, cx0, cy0, cx1, cy1 = crop_key
        crop_sz = cx1 - cx0
    else:
        if max(bw, bh) < 40.0:
            crop_sz = 256
        else:
            crop_sz = min(min(h, w), max(256, int(math.ceil(max(bw, bh) * 3.0 / 32.0)) * 32))
            if crop_sz >= int(0.75 * min(h, w)):
                return None
        cx0 = max(0, min(w - crop_sz, int(round((bx0 + bx1) / 2.0 - crop_sz / 2.0))))
        cy0 = max(0, min(h - crop_sz, int(round((by0 + by1) / 2.0 - crop_sz / 2.0))))
        cx0 = max(0, min(w - crop_sz, (cx0 // 4) * 4))
        cy0 = max(0, min(h - crop_sz, (cy0 // 4) * 4))
        cx1, cy1 = cx0 + crop_sz, cy0 + crop_sz
    # Unconditional bounds check before slicing or indexing
    if crop_sz <= 0 or cx0 < 0 or cy0 < 0 or cx1 > w or cy1 > h or (cy1 - cy0) != crop_sz:
        return None
    if any(x < cx0 or x >= cx1 or y < cy0 or y >= cy1 for x, y in pts):
        return None
    key = (image_id, cx0, cy0, cx1, cy1)
    multi = len(pts) == 1
    with torch.inference_mode():
        if key in _crop_emb:
            cemb = _crop_emb[key]
            _crop_emb.move_to_end(key)
            if dev in ("cuda", "mps"):
                cemb = _move_emb(cemb, dev, dt)
                _crop_emb[key] = cemb
                _trim_hot_crops(dev)
        else:
            if crop_key is not None:
                return None
            crop_hi = cv2.resize(img[cy0:cy1, cx0:cx1], (int(size), int(size)), interpolation=cv2.INTER_LINEAR)
            pv = _preprocess(proc, crop_hi, size, dev, dt)
            cemb = model.get_image_embeddings(pv)
            _crop_emb[key] = cemb
            while len(_crop_emb) > CACHE_CROPS:
                _crop_emb.popitem(last=False)
            _trim_hot_crops(dev)
        cpts = pts - np.array([cx0, cy0], dtype=np.float32)
        cscaled = cpts * np.array([size / crop_sz, size / crop_sz], dtype=np.float32) - 0.5
        cout = model(image_embeddings=cemb,
                     input_points=torch.from_numpy(cscaled)[None, None].to(dev, torch.float32),
                     input_labels=torch.from_numpy(lbl)[None, None].to(dev),
                     multimask_output=multi)
        need_lg = outline != "pixel"
        cmasks, clogits = _post_process(proc, cout.pred_masks, crop_sz, crop_sz, need_logits=need_lg)
        cpred = cout.pred_masks.float().cpu()
        cscores = cout.iou_scores[0, 0].float().cpu().numpy()
    v_crop = valid[cy0:cy1, cx0:cx1]
    coarse_crop = coarse_mask[cy0:cy1, cx0:cx1] if coarse_mask is not None else None
    max_zdim = min(crop_sz * 0.65, max(128.0, 2.2 * max(bw, bh)))
    best = None
    for j in np.argsort(-cscores):
        cm = cmasks[j] & v_crop
        if coarse_crop is not None and max(bw, bh) >= 40.0:
            inter = int((cm & coarse_crop).sum())
            union = int((cm | coarse_crop).sum())
            if union == 0 or inter / union < 0.45:
                continue
        cpolys = (logits_to_polygons(clogits[j], v_crop, min_area)
                  if (outline != "pixel" and clogits is not None)
                  else mask_to_polygons(cm, min_area, outline))
        if not cpolys:
            continue
        polys = [[[[x + cx0, y + cy0] for x, y in r] for r in rings] for rings in cpolys]
        kept = keep_clicked(polys, pts, lbl)
        covers = sum(any(_poly_dist(rings, x, y) <= 0 for rings in kept) for x, y in pos_pts)
        if covers == 0 or covers < orig_covers:
            continue
        z_neg = sum(any(_poly_dist(rings, x, y) <= 0 for rings in kept) for x, y in neg_pts)
        if z_neg > orig_neg_viol:
            continue
        zx0, zy0, zx1, zy1 = _poly_bbox(kept)
        zbw, zbh = zx1 - zx0, zy1 - zy0
        if zbw * zbh > max(4.0 * bw * bh, 100.0) or max(zbw, zbh) >= max_zdim:
            continue
        if ((cx0 > 0 and zx0 <= cx0 + 0.5) or (cx1 < w and zx1 >= cx1 - 0.5)
                or (cy0 > 0 and zy0 <= cy0 + 0.5) or (cy1 < h and zy1 >= cy1 - 0.5)):
            continue
        cand = (covers, z_neg, float(cscores[j]), kept, j, cm)
        if best is None or (-cand[0], cand[1], -cand[2]) < (-best[0], best[1], -best[2]):
            best = cand
    if best is None:
        return None
    covers, z_neg, sc, kept, j, _ = best
    merged_lr = orig_lr.detach().float().cpu().clone()
    h_lr, w_lr = merged_lr.shape[-2:]
    ly0 = min(h_lr - 1, max(0, int(round(cy0 * h_lr / h))))
    ly1 = min(h_lr, max(ly0 + 1, int(round(cy1 * h_lr / h))))
    lx0 = min(w_lr - 1, max(0, int(round(cx0 * w_lr / w))))
    lx1 = min(w_lr, max(lx0 + 1, int(round(cx1 * w_lr / w))))
    z_lr = F.interpolate(cpred[:, :, j, :, :], size=(ly1 - ly0, lx1 - lx0), mode="bilinear", align_corners=False)
    merged_lr[:, :, ly0:ly1, lx0:lx1] = z_lr
    return (covers, z_neg, sc, kept, merged_lr, None)


# ---------------------------------------------------------------- job
def encode_image(req, pixels):
    """Pre-warm image embeddings on GPU/MPS so the next click skips the ViT encoder."""
    if pixels is None:
        return {"encoded": False, "error": "no pixels"}
    proc, model = load()
    dev, dt = S["device"], model.dtype
    if dev == "cpu":
        return {"encoded": False, "skipped": "cpu"}
    image_id = req.get("image_id")
    if not image_id:
        raise ValueError("image_id is required")
    size = float(model.config.prompt_encoder_config.image_size)
    with torch.inference_mode():
        if image_id in _emb:
            entry = _emb[image_id]
            emb, hw, valid = entry[:3]
            img = entry[3] if len(entry) > 3 else None
            _emb.move_to_end(image_id)
            if dev in ("cuda", "mps"):
                emb = _move_emb(emb, dev, dt)
                _emb[image_id] = (emb, hw, valid, img)
                _trim_hot_emb(dev)
            return {"encoded": True, "cached": True}
        (w, h), n = req["size"], len(pixels)
        if not 0 < w * h <= MAX_PIXELS or n != w * h * 4:
            raise ValueError(f"the image must be {w} x {h} x 4 bytes (RGBA), at most {MAX_PIXELS} px; got {n}")
        a = np.frombuffer(pixels, np.uint8).reshape(h, w, 4)
        valid = a[..., 3] > 0
        if not valid.any():
            return {"encoded": False, "error": "No imagery here"}
        img = a[..., :3].copy()
        if not valid.all():
            img[~valid] = _nodata_fill(proc)
        pv = _preprocess(proc, img, size, dev, dt)
        emb = model.get_image_embeddings(pv)
        _emb[image_id] = (emb, (h, w), valid, img)
        while len(_emb) > CACHE_IMAGES:
            old_id, _ = _emb.popitem(last=False)
            for k in [k for k in _crop_emb if k[0] == old_id]:
                del _crop_emb[k]
        _trim_hot_emb(dev)
    return {"encoded": True}


def predict(req, pixels=None):
    """run_predict; if the GPU fails (out of memory, an op MPS lacks, ..) move to the CPU and retry once."""
    try:
        return run_predict(req, pixels)
    except RuntimeError as e:                     # torch's OutOfMemoryError and NotImplementedError too
        if S["device"] == "cpu" or S["model"] is None:      # load() handles its own GPU failure
            raise
        use_cpu(e)
    return run_predict(req, pixels)


def run_predict(req, pixels):
    if "image" in req:                            # the request format before API 2
        raise ValueError("this engine is from a newer Segmentot than the plugin that called it: restart QGIS")
    pts = np.array(req["points"], dtype=np.float32)
    lbl = np.array(req["labels"], dtype=np.int64)
    # checked here so a bad request cannot fail inside torch and pass for a GPU failure
    if not len(pts) or pts.shape[1:] != (2,) or lbl.shape != pts.shape[:1] or not np.isin(lbl, (0, 1)).all():
        raise ValueError("points must be [[x, y], ..] with one label (1 or 0) each")
    proc, model = load()
    dev, dt = S["device"], model.dtype
    image_id = req.get("image_id")
    is_hover = bool(req.get("hover"))
    is_progressive = bool(req.get("progressive")) and not is_hover
    is_refine_zoom = bool(req.get("refine_zoom")) and not is_hover
    pos_pts = [(float(x), float(y)) for (x, y), lb in zip(pts, lbl) if lb == 1]
    neg_pts = [(float(x), float(y)) for (x, y), lb in zip(pts, lbl) if lb == 0]
    size = float(model.config.prompt_encoder_config.image_size)      # 1024; 1008 for SAM 3
    min_area = float(req.get("min_area_px", 4))
    outline = req.get("outline", "pixel")
    pts_key = tuple((round(float(x), 3), round(float(y), 3)) for x, y in pts)
    lbl_key = tuple(int(v) for v in lbl)
    with torch.inference_mode():
        if image_id in _emb:
            entry = _emb[image_id]
            emb, (h, w), valid = entry[:3]
            img = entry[3] if len(entry) > 3 else None
            _emb.move_to_end(image_id)
            if dev in ("cuda", "mps"):
                emb = _move_emb(emb, dev, dt)
                _emb[image_id] = (emb, (h, w), valid, img)
                _trim_hot_emb(dev)
        else:
            if pixels is None:
                return {"need_image": True}
            (w, h), n = req["size"], len(pixels)
            if not 0 < w * h <= MAX_PIXELS or n != w * h * 4:
                raise ValueError(f"the image must be {w} x {h} x 4 bytes (RGBA), at most {MAX_PIXELS} px; got {n}")
            a = np.frombuffer(pixels, np.uint8).reshape(h, w, 4)
            valid = a[..., 3] > 0
            if not valid.any():
                return {"candidates": [], "error": "No imagery here"}
            if valid.mean() < 0.01:
                pos_rc = [(min(h - 1, max(0, int(y))), min(w - 1, max(0, int(x)))) for x, y in pos_pts]
                if not any(valid[max(0, r - 16):r + 17, max(0, c - 16):c + 17].any() for r, c in pos_rc):
                    return {"candidates": [], "error": "No imagery here"}
            img = a[..., :3].copy()
            if not valid.all():                   # neutral model-mean fill stops no-data border bleed
                img[~valid] = _nodata_fill(proc)
            pv = _preprocess(proc, img, size, dev, dt)
            emb = model.get_image_embeddings(pv)
            _emb[image_id] = (emb, (h, w), valid, img)
            while len(_emb) > CACHE_IMAGES:
                old_id, _ = _emb.popitem(last=False)
                for k in [k for k in _crop_emb if k[0] == old_id]:
                    del _crop_emb[k]
            _trim_hot_emb(dev)
        valid_all = bool(valid.all())
        cands = None
        if is_refine_zoom and _last_coarse[0] is not None:
            lc = _last_coarse[0]
            if lc[:5] == (image_id, pts_key, lbl_key, outline, min_area):
                cands = list(lc[5])
                _last_coarse[0] = None
        if cands is None:
            # the model adds 0.5 to a point (it takes pixel indices); ours are continuous: pixel i spans i..i+1
            scaled = pts * np.array([size / w, size / h], dtype=np.float32) - 0.5
            multi = len(pts) == 1
            in_mask = None
            if not multi:
                mask_id = req.get("mask_id")
                if mask_id in _masks and _masks[mask_id][0] == image_id:
                    _masks.move_to_end(mask_id)
                    in_mask = _masks[mask_id][1].to(dev, dt)
            kw = {"input_masks": in_mask} if in_mask is not None else {}
            t_pts = torch.from_numpy(scaled)[None, None].to(dev, torch.float32)
            t_lbl = torch.from_numpy(lbl)[None, None].to(dev)
            need_lg = outline != "pixel"

            def _decode(extra):
                out = model(image_embeddings=emb, input_points=t_pts, input_labels=t_lbl,
                            multimask_output=multi, **extra)
                sc = out.iou_scores[0, 0].float().cpu().numpy()
                # 1-candidate hover fast path: rank on-device, transfer & polygonize only the winner
                if is_hover and _is_hf_proc(proc) and out.pred_masks.ndim == 5:
                    up = F.interpolate(out.pred_masks[0, 0].unsqueeze(0).float(), size=(h, w),
                                       mode="bilinear", align_corners=False)[0]
                    if pos_pts:
                        r0 = min(h - 1, max(0, int(pos_pts[0][1])))
                        c0 = min(w - 1, max(0, int(pos_pts[0][0])))
                        hits = (up[:, r0, c0] > 0).cpu().numpy()
                    else:
                        hits = np.ones(len(sc), dtype=bool)
                    order = sorted(range(len(sc)), key=lambda i: (-int(hits[i]), -float(sc[i])))
                    for i in order:
                        if need_lg:
                            lg_i = up[i].cpu().numpy()
                            polys = logits_to_polygons(lg_i, valid, min_area)
                        else:
                            m_i = (up[i] > 0).cpu().numpy()
                            if not valid_all:
                                m_i = m_i & valid
                            polys = mask_to_polygons(m_i, min_area, outline)
                        if polys:
                            kept = keep_clicked(polys, pts, lbl)
                            covers = sum(any(_poly_dist(rings, x, y) <= 0 for rings in kept) for x, y in pos_pts)
                            return [(covers, 0, float(sc[i]), kept, None, None)]
                    return []

                ms, lg = _post_process(proc, out.pred_masks, h, w, need_logits=need_lg)
                pm = out.pred_masks.float().cpu()
                out_cands = []
                for i in np.argsort(-sc):
                    m = ms[i] if valid_all else (ms[i] & valid)
                    polys = (logits_to_polygons(lg[i], valid, min_area)
                             if (need_lg and lg is not None)
                             else mask_to_polygons(m, min_area, outline))
                    if polys:
                        kept = keep_clicked(polys, pts, lbl)
                        covers = sum(any(_poly_dist(rings, x, y) <= 0 for rings in kept) for x, y in pos_pts)
                        neg_v = sum(any(_poly_dist(rings, x, y) <= 0 for rings in kept) for x, y in neg_pts)
                        out_cands.append((covers, neg_v, float(sc[i]), kept, pm[:, :, i, :, :], m))
                out_cands.sort(key=lambda o: (-o[0], o[1], -o[2]))
                return out_cands

            cands = _decode(kw)
            if in_mask is not None and (
                    not cands or cands[0][0] < len(pos_pts) or cands[0][1] > 0 or cands[0][2] < 0.2):
                alt = _decode({})
                if alt and (not cands
                            or (-alt[0][0], alt[0][1], -alt[0][2]) < (-cands[0][0], cands[0][1], -cands[0][2])):
                    cands = alt
    if not is_hover and img is not None and min(h, w) >= 512 and cands and cands[0][0] > 0:
        bx0, by0, bx1, by1 = _poly_bbox(cands[0][3])
        max_dim = max(bx1 - bx0, by1 - by0)
        if len(pts) == 1:
            mid_low = (S["model_id"] or "").lower()
            fast_model = "edgetam" in mid_low or "tiny" in mid_low
            if dev == "cpu":
                want_zoom = 2.0 <= max_dim < 64.0
            elif fast_model:
                want_zoom = 2.0 <= max_dim < 144.0
            else:
                want_zoom = 2.0 <= max_dim < 80.0 or (max_dim < 144.0 and cands[0][2] < 0.92)
            if want_zoom:
                ckey = _zoom_crop_key(image_id, (bx0, by0, bx1, by1), h, w)
                if ckey is not None:
                    if is_progressive and not is_refine_zoom and ckey not in _crop_emb:
                        _last_coarse[0] = (image_id, pts_key, lbl_key, outline, min_area, list(cands), ckey)
                        return {
                            "candidates": [
                                {"polygons": p, "score": round(sc, 4), "mask_id": _put_mask(image_id, lr)}
                                for _, _, sc, p, lr, _ in cands
                            ],
                            "needs_zoom": True,
                            "image_size": model_image_size(),
                        }
                    refined = _zoom_refine(
                        proc, model, dev, dt, size, image_id, img, valid,
                        pts, lbl, pos_pts, neg_pts, outline, min_area,
                        (bx0, by0, bx1, by1), cands[0][4],
                        cands[0][0], cands[0][1], coarse_mask=cands[0][5],
                    )
                    if refined is not None:
                        cands[0] = refined
        else:
            cached_key = _find_cached_crop(image_id, (bx0, by0, bx1, by1), pts, h, w)
            if cached_key is not None:
                refined = _zoom_refine(
                    proc, model, dev, dt, size, image_id, img, valid,
                    pts, lbl, pos_pts, neg_pts, outline, min_area,
                    (bx0, by0, bx1, by1), cands[0][4],
                    cands[0][0], cands[0][1], coarse_mask=cands[0][5], crop_key=cached_key,
                )
                if refined is not None:
                    cands[0] = refined
    if is_hover:
        return {
            "candidates": [{"polygons": p, "score": round(sc, 4)} for _, _, sc, p, _, _ in cands],
            "image_size": model_image_size(),
        }
    return {
        "candidates": [{"polygons": p, "score": round(sc, 4), "mask_id": _put_mask(image_id, lr)}
                       for _, _, sc, p, lr, _ in cands],
        "image_size": model_image_size(),
    }


# ---------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj, separators=(",", ":")).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if code >= 400:
            self.close_connection = True
            self.send_header("Connection", "close")
        nonce = self.headers.get("X-Segmentot-Nonce")
        if KEY and nonce:           # proof for the plugin that this is its engine, not another program on the port
            self.send_header("X-Segmentot-Proof", hmac.new(
                KEY, f"{self.server.server_port}:{nonce}".encode(), "sha256").hexdigest())
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
            res = {"ok": True, "engine": "segmentot", "api": API, "device": S["device"], "gpu": gpu_name(),
                   "loaded": S["model"] is not None, "model": S["model_id"].split("/")[-1],
                   "image_size": model_image_size(), "torch": torch.__version__}
            if S["notice"]:
                res["notice"] = S["notice"]
            self._send(200, res)
        except Exception as e:
            log(traceback.format_exc())
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        global _last_request, _hover_gen, _pending_clicks
        if self._refused():
            return
        _last_request = time.time()
        if self.path not in ("/predict", "/encode"):
            return self._send(404, {"error": "not found"})
        tracked_click = False
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n < 0:
                return self._send(400, {"error": "invalid Content-Length"})
            if n > MAX_BODY:
                return self._send(413, {"error": "request too large"})
            body, head = self.rfile.read(n), self.headers.get("X-Segmentot-Request")
            # with an image the JSON is in this header and the body is the pixels
            req, pixels = (json.loads(head), body) if head else (json.loads(body), None)
            if self.path == "/encode":
                with _gen_lock:
                    busy_click = _pending_clicks > 0
                if busy_click:
                    return self._send(200, {
                        "encoded": False, "stale": True, "device": S["device"],
                        "image_size": model_image_size(), "seconds": 0.0,
                    })
                t = time.time()
                stale_enc = False
                with _lock:
                    with _gen_lock:
                        if _pending_clicks > 0:
                            stale_enc = True
                    if not stale_enc:
                        try:
                            res = encode_image(req, pixels)
                        except RuntimeError as e:
                            if S["device"] == "cpu" or S["model"] is None:
                                raise
                            use_cpu(e)
                            res = {"encoded": False, "skipped": "cpu"}
                        res["device"] = S["device"]
                        res["image_size"] = model_image_size()
                        if S["notice"]:
                            res["notice"] = S["notice"]
                if stale_enc:
                    return self._send(200, {
                        "encoded": False, "stale": True, "device": S["device"],
                        "image_size": model_image_size(), "seconds": 0.0,
                    })
                res["seconds"] = round(time.time() - t, 3)
                return self._send(200, res)
            is_hover = bool(req.get("hover"))
            my_gen = 0
            with _gen_lock:
                if is_hover:
                    _hover_gen += 1
                    my_gen = _hover_gen
                    stale = _pending_clicks > 0
                else:
                    _pending_clicks += 1
                    tracked_click = True
                    stale = False
            if is_hover and stale:
                return self._send(200, {
                    "candidates": [], "stale": True, "device": S["device"],
                    "image_size": model_image_size(), "seconds": 0.0,
                })
            t = time.time()
            stale_hover = False
            with _lock:
                if is_hover:
                    with _gen_lock:
                        if _pending_clicks > 0 or my_gen != _hover_gen:
                            stale_hover = True
                if not stale_hover:
                    res = predict(req, pixels)
                    res["device"] = S["device"]
                    res["image_size"] = model_image_size()
                    if S["notice"]:
                        res["notice"] = S["notice"]
            if stale_hover:
                return self._send(200, {
                    "candidates": [], "stale": True, "device": S["device"],
                    "image_size": model_image_size(), "seconds": 0.0,
                })
            res["seconds"] = round(time.time() - t, 3)
            self._send(200, res)
        except Exception as e:
            log(traceback.format_exc())
            self._send(500, {"error": f"{type(e).__name__}: {e}"})
        finally:
            if tracked_click:
                with _gen_lock:
                    _pending_clicks = max(0, _pending_clicks - 1)
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
