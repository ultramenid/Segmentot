"""Real-model GPU -> CPU fallback: load SAM2.1 on the GPU, segment, inject a GPU failure, check the CPU takes over."""
import importlib.util
import os
import sys
import time

import numpy as np

# run with the engine's own Python (<engine folder>/venv/bin/python): the model is read from that folder, offline
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(sys.executable))), "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
SERVER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server", "segmentot_server.py")
spec = importlib.util.spec_from_file_location("srv", SERVER)
srv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(srv)
srv.S["model_id"], srv.S["device"] = "facebook/sam2.1-hiera-base-plus", srv.pick_device("auto")
start = srv.S["device"]
print("device picked first:", start, flush=True)
t = time.time()
proc, model = srv.load()
print("model loaded on %s in %.1f s" % (srv.S["device"], time.time() - t), flush=True)

rng = np.random.default_rng(1)
img = np.zeros((1024, 1024, 4), np.uint8)
img[..., 3] = 255
img[..., :3] = np.clip(np.array([70, 110, 60]) + rng.normal(0, 6, (1024, 1024, 3)), 1, 255)
img[400:520, 300:500, :3] = (200, 60, 50)  # truth: 200 x 120 px = 24000 px2
px = img.tobytes()
req = {"image_id": "a", "size": [1024, 1024], "points": [[400.0, 460.0]], "labels": [1]}


def area(res):
    ring = np.asarray(res["candidates"][0]["polygons"][0][0])
    return 0.5 * abs(np.dot(ring[:-1, 0], ring[1:, 1]) - np.dot(ring[1:, 0], ring[:-1, 1]))


ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS  " if cond else "FAIL  ") + name + ("  — " + str(detail) if detail else ""), flush=True)


t = time.time()
r1 = srv.predict(dict(req), px)
t1 = time.time() - t
check(
    "GPU run gives the rectangle",
    abs(area(r1) - 24000) / 24000 < 0.03 and srv.S["device"] == start,
    "%.0f px2 in %.2f s on %s" % (area(r1), t1, srv.S["device"]),
)
if start == "cpu":
    print("no GPU on this computer: nothing to fall back from")
    sys.exit(0 if ok else 1)

orig = model.forward


def boom(*a, **k):
    if srv.S["device"] != "cpu":
        raise RuntimeError("MPS backend out of memory (injected by the test)")
    return orig(*a, **k)


model.forward = boom
r2 = srv.predict(dict(req))  # image is cached -> fails on the GPU
check("GPU failure switches the engine to the CPU", srv.S["device"] == "cpu" and r2 == {"need_image": True}, r2)
check("notice tells the user", bool(srv.S["notice"]) and "CPU" in srv.S["notice"], srv.S["notice"])
check("model weights really moved to the CPU, in float32",
      next(model.parameters()).device.type == "cpu" and model.dtype == srv.torch.float32)
t = time.time()
r3 = srv.predict(dict(req), px)
t3 = time.time() - t
check(
    "the resent request succeeds on the CPU with the same outline",
    abs(area(r3) - area(r1)) / area(r1) < 0.02,
    "%.0f px2 in %.2f s" % (area(r3), t3),
)
t = time.time()
r4 = srv.predict(dict(req, points=[[400.0, 460.0], [900.0, 900.0]], labels=[1, 0]))
t4 = time.time() - t
check(
    "later clicks keep working on the CPU", abs(area(r4) - 24000) / 24000 < 0.05, "%.0f px2 in %.2f s" % (area(r4), t4)
)
sys.exit(0 if ok else 1)
