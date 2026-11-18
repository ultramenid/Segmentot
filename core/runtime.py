"""Segmentot engine environment: hardware detection, one-time setup, and the engine process.

Everything lives in one folder (SEGMENTOT_HOME or ~/.segmentot, %LOCALAPPDATA%\\Segmentot on Windows):
  bin/uv        uv package manager (downloaded once)
  python/       private Python 3.12 managed by uv (independent of QGIS's Python)
  venv/         engine environment: torch (CUDA / Apple MPS / CPU, chosen by uv), transformers ...
  hf/           model weights: SAM2.1, EdgeTAM (Hugging Face cache)
  env.json      what was installed, for which hardware
  engine.key    secret the engine proves it knows, so the plugin never talks to another program on its port
  engine.log    engine output
No QGIS imports: safe to use from background tasks and threads.
"""
import getpass
import glob
import hashlib
import io
import json
import os
import platform
import re
import secrets
import shutil
import subprocess  # running its own engine (uv, python) is what this module is for  # nosec B404
import tarfile
import tempfile
import threading
import time
import urllib.request
import zipfile

from .edition import EDITION

PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # this file is in core/
EDITION_LABEL = {"cpu": "CPU edition", "gpu": "GPU edition"}.get(EDITION, "")
SERVER = os.path.join(PLUGIN_DIR, "server", "segmentot_server.py")
ENV_VERSION = 1                      # bump when the engine's requirements change
PYTHON_VERSION = "3.12"
PACKAGES = ["transformers>=4.57,<6", "huggingface_hub", "safetensors", "pillow", "numpy",
            "opencv-python-headless", "rasterio", "timm"]        # timm: EdgeTAM's image encoder
MODELS = {                           # key -> (Hugging Face id, label, download size in MB)
    "edge": ("yonigozlan/EdgeTAM-hf", "EdgeTAM (fastest, smallest)", 56),
    "tiny": ("facebook/sam2.1-hiera-tiny", "SAM 2.1 Tiny (CPU friendly)", 156),
    "small": ("facebook/sam2.1-hiera-small", "SAM 2.1 Small", 184),
    "base": ("facebook/sam2.1-hiera-base-plus", "SAM 2.1 Base+ (recommended with a GPU)", 323),
    "large": ("facebook/sam2.1-hiera-large", "SAM 2.1 Large (needs a strong GPU)", 898),
    "sam3": ("facebook/sam3", "SAM 3 (newest, needs approval)", 3440),
}
GATED = ("sam3",)                    # Hugging Face gives these only to accounts their maker approved
WIN = os.name == "nt"


def model_docs(model_key):
    """The model's own page: what it is, how it was made, its paper and its license."""
    return "https://huggingface.co/" + MODELS[model_key][0]


def size_text(mb):
    return "{:.1f} GB".format(mb / 1000) if mb >= 1000 else "{} MB".format(mb)


def token_path():
    """Where huggingface_hub looks for the access token (HF_HOME is the engine's hf folder)."""
    return os.path.join(home(), "hf", "token")


def save_hf_token(token):
    os.makedirs(os.path.dirname(token_path()), exist_ok=True)
    fd = os.open(token_path(), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)      # readable by this user only
    with os.fdopen(fd, "w") as f:
        f.write(token.strip())


def engine_key(create=False):
    """The secret our engine proves it knows in every answer (engine.py checks it). None when there is none;
    start_engine() makes it. Readable by this user only, like the access token."""
    path = os.path.join(home(), "engine.key")
    try:
        with open(path) as f:
            key = f.read().strip()
        if key or not create:
            return key or None
        os.remove(path)                      # left empty by a crash
    except OSError:
        if not create:
            return None
    os.makedirs(home(), exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:                  # another QGIS made it just now
        return engine_key()
    key = secrets.token_hex(32)
    with os.fdopen(fd, "w") as f:
        f.write(key)
    return key


NO_WINDOW = 0x08000000 if WIN else 0     # CREATE_NO_WINDOW
# Bump only when setup() needs a newer uv; copy each hash from that release's uv-<target>.<ext>.sha256 asset.
UV_VERSION = "0.13.7"
UV_SHA256 = {
    "x86_64-pc-windows-msvc": "0cdc5b38fe5a8168ed6a25814f5ba43c21d313dd3187314fccb37b7cec46e021",
    "aarch64-pc-windows-msvc": "505d9e65e7692bac3ddf98c66754aa26c5050c409c184af30a881a0a72cc85e3",
    "x86_64-apple-darwin": "20c47e4c1f630b8abaff8c291fa9b9aa300fef92c4fbff95adf198b55287cda7",
    "aarch64-apple-darwin": "81b811b57e4624df7ed3aac148e0359100d9962f3d9dce00aec533dc2e086731",
    "x86_64-unknown-linux-gnu": "c5221ff41b7892bfc223543883600125f6a47fe845093e0f5bdff4f359196d83",
    "aarch64-unknown-linux-gnu": "71279f8d3c67241e807b9d8db872d1bf541579fb9f08b7d1eb1682196d1e57dc",
}


class SetupError(Exception):
    pass


# ------------------------------------------------------------------ paths & state
def home():
    d = os.environ.get("SEGMENTOT_HOME")
    if not d:
        d = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "Segmentot") if WIN \
            else os.path.expanduser("~/.segmentot")
    return d


def venv_python():
    v = os.path.join(home(), "venv")
    return os.path.join(v, "Scripts", "python.exe") if WIN else os.path.join(v, "bin", "python")


def uv_path():
    return os.path.join(home(), "bin", "uv.exe" if WIN else "uv")


def load_state():
    try:
        with open(os.path.join(home(), "env.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(st):
    os.makedirs(home(), exist_ok=True)
    with open(os.path.join(home(), "env.json"), "w") as f:
        json.dump(st, f, indent=2)


def is_ready():
    st = load_state()
    return st.get("env_version") == ENV_VERSION and st.get("complete") and os.path.exists(venv_python())


def home_writable():
    """Can the engine folder be created and written? If not, the user needs permission_guide()."""
    try:
        os.makedirs(home(), exist_ok=True)
        with tempfile.TemporaryFile(dir=home()):
            return True
    except OSError:
        return False


def permission_guide():
    """Ready-to-paste fixes for an install that needs administrator rights: [(title, how to run it, command)]."""
    h = home()
    if WIN:
        user = "\\".join(x for x in (os.environ.get("USERDOMAIN"), os.environ.get("USERNAME")) if x)
        how = "Start menu → type PowerShell → right-click it → Run as administrator, then paste:"
        return [
            ("Give your account access to the engine folder", how,
             'New-Item -ItemType Directory -Force "{0}" | Out-Null\n'
             'icacls "{0}" /grant "{1}:(OI)(CI)F" /T'.format(h, user)),
            ("Allow long file paths (when the log says a path is too long or a file cannot be found)",
             how + " Then restart the computer.",
             'New-ItemProperty -Path "HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem" '
             '-Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force'),
        ]
    items = [("Give your account access to the engine folder",
              "Open Terminal, paste this and type your password when it asks:",
              'sudo mkdir -p "{0}" && sudo chown -R "{1}" "{0}"'.format(h, getpass.getuser()))]
    if os.path.exists("/.flatpak-info"):
        items.append(("Let the Flatpak QGIS use the NVIDIA GPU", "Open a terminal, paste this, then restart QGIS:",
                      "flatpak override --user --device=all org.qgis.qgis"))
    return items


def clean_env(extra=None):
    """Environment for child processes, without QGIS's Python/GDAL/library settings."""
    env = dict(os.environ)
    for k in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE", "PYTHONEXECUTABLE",
              "VIRTUAL_ENV", "CONDA_PREFIX", "LD_LIBRARY_PATH", "LD_PRELOAD", "DYLD_LIBRARY_PATH",
              "DYLD_FALLBACK_LIBRARY_PATH", "GDAL_DATA", "GDAL_DRIVER_PATH", "PROJ_LIB", "PROJ_DATA",
              "QT_PLUGIN_PATH", "SSL_CERT_FILE"):
        env.pop(k, None)
    h = home()
    env.update({"UV_PYTHON_INSTALL_DIR": os.path.join(h, "python"), "UV_CACHE_DIR": os.path.join(h, "cache"),
                "UV_PYTHON_PREFERENCE": "only-managed", "UV_NO_CONFIG": "1",
                "HF_HOME": os.path.join(h, "hf"), "HF_HUB_DISABLE_TELEMETRY": "1",
                "PYTHONNOUSERSITE": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                "PYTORCH_ENABLE_MPS_FALLBACK": "1"})
    env.update(extra or {})
    return env


# ------------------------------------------------------------------ hardware
def _run_quiet(cmd, timeout=15):
    try:
        # safe for bandit B603: a fixed program (nvidia-smi) with fixed arguments, no shell
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, creationflags=NO_WINDOW)  # nosec B603
        return r.stdout if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def detect():
    """What this computer has, and the plan for it."""
    system, arch = platform.system(), platform.machine().lower()
    info = {"os": system, "arch": arch, "flatpak": os.path.exists("/.flatpak-info"),
            "nvidia": None, "nvidia_device": False, "apple_gpu": False}
    smi = shutil.which("nvidia-smi") or (r"C:\Windows\System32\nvidia-smi.exe" if WIN else None)
    out = _run_quiet([smi, "--query-gpu=name,memory.total,driver_version",
                      "--format=csv,noheader,nounits"]) if smi else ""
    if out.strip():
        name, mem, drv = [x.strip() for x in out.strip().splitlines()[0].split(",")[:3]]
        try:
            vram = round(float(mem) / 1024, 1)
        except ValueError:                   # some cards report "[N/A]"
            vram = None
        info["nvidia"] = {"name": name, "vram_gb": vram, "driver": drv}
        cc = _run_quiet([smi, "--query-gpu=compute_cap", "--format=csv,noheader"]).strip()
        try:
            info["nvidia"]["compute_cap"] = float(cc.splitlines()[0])
        except (ValueError, IndexError):
            pass
    elif system == "Linux":
        for p in ("/sys/module/nvidia/version", "/proc/driver/nvidia/version"):
            try:
                with open(p) as f:
                    m = re.search(r"\b(\d{3,}\.\d+(?:\.\d+)?)\b", f.read())
                if m:
                    info["nvidia"] = {"name": "NVIDIA GPU", "vram_gb": None, "driver": m.group(1)}
                    break
            except OSError:
                pass
    if info["nvidia"]:
        info["nvidia_device"] = WIN or system != "Linux" or os.path.exists("/dev/nvidiactl")
        info["libcuda"] = _libcuda_loads()
    info["apple_gpu"] = system == "Darwin" and arch in ("arm64", "aarch64")

    if EDITION == "cpu":
        accel = "cpu"
    elif info["nvidia"] and info["nvidia_device"] and info["libcuda"]:
        accel = "cuda"
    elif info["apple_gpu"]:
        accel = "mps"
    else:
        accel = "cpu"
    info["accel"] = accel
    # uv picks the newest CUDA build this driver supports
    info["torch_backend"] = "auto" if accel == "cuda" else ("cpu" if system in ("Linux", "Windows") else None)
    cc = (info["nvidia"] or {}).get("compute_cap")
    if accel == "cuda" and cc is not None and cc < 7.5:
        # CUDA 13 builds dropped Maxwell/Pascal/Volta (GTX 9xx/10xx, Titan V): use the CUDA 12.6 build
        info["torch_backend"] = "cu126"
    info["model"] = "base" if accel in ("cuda", "mps") else "small"
    notes = []
    info["edition"] = EDITION
    info["gpu_missing"] = EDITION == "gpu" and accel == "cpu"
    if EDITION == "cpu":
        notes.append("CPU edition: runs on the processor on every computer (no GPU needed). "
                     "Each new view takes a few seconds; clicks are quick.")
        info["notes"] = notes
        return info
    if info["gpu_missing"]:
        notes.append("GPU edition, but no supported GPU was found (it needs an NVIDIA card with a recent "
                     "driver, or an Apple M-series Mac). The CPU edition is the better fit here; "
                     "if you continue, this one also runs on the CPU.")
    if info["nvidia"] and accel != "cuda":
        drv = info["nvidia"]["driver"]
        if info["flatpak"] and not info["nvidia_device"]:
            notes.append("This QGIS is a Flatpak and cannot see the NVIDIA GPU. To use it, run\n"
                         "  flatpak override --user --device=all org.qgis.qgis\n"
                         "restart QGIS and choose Settings… → Reinstall… in the Segmentot panel. "
                         "Until then Segmentot runs on the CPU.")
        elif info["flatpak"]:
            notes.append("This QGIS is a Flatpak without the NVIDIA driver extension for driver {0}. "
                         "To use the GPU, run\n  flatpak install flathub org.freedesktop.Platform.GL.nvidia-{1}\n"
                         "(or simply: flatpak update), restart QGIS and choose Settings… → Reinstall… "
                         "in the Segmentot panel. "
                         "Until then Segmentot runs on the CPU.".format(drv, drv.replace(".", "-")))
        else:
            notes.append("An NVIDIA driver ({}) is present but CUDA cannot be loaded; running on the CPU.".format(drv))
    if accel == "cpu" and not info["gpu_missing"]:
        notes.append("No usable GPU: running on the CPU. Each new view takes a few seconds; clicks are quick.")
    info["notes"] = notes
    return info


def _libcuda_loads():
    import ctypes
    for name in (("nvcuda.dll",) if WIN else ("libcuda.so.1", "libcuda.so")):
        try:
            ctypes.CDLL(name)
            return True
        except OSError:
            pass
    return False


def describe(info):
    if info["accel"] == "cuda":
        n = info["nvidia"]
        name = n["name"] if n["name"].upper().startswith("NVIDIA") else "NVIDIA " + n["name"]
        hw = "{} ({}driver {})".format(name, "{} GB, ".format(n["vram_gb"]) if n["vram_gb"] else "", n["driver"])
    elif info["accel"] == "mps":
        hw = "Apple Silicon GPU (Metal)"
    else:
        hw = "CPU only"
    return "{} {} · {}{}".format(info["os"], info["arch"], hw, " · Flatpak" if info["flatpak"] else "")


# ------------------------------------------------------------------ setup steps
def _uv_target():
    system, arch = platform.system(), platform.machine().lower()
    a = "aarch64" if arch in ("arm64", "aarch64") else "x86_64"
    if system == "Windows":
        return a + "-pc-windows-msvc", "zip"
    if system == "Darwin":
        return a + "-apple-darwin", "tar.gz"
    return a + "-unknown-linux-gnu", "tar.gz"


def _download(url, log, timeout=120):
    if not url.startswith("https://"):
        raise SetupError("Refusing to download from " + url)
    log("Downloading " + url)
    req = urllib.request.Request(url, headers={"User-Agent": "Segmentot-QGIS-plugin"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # https only, checked above  # nosec B310
        return r.read()


def get_uv(log):
    dst = uv_path()
    if _run_quiet([dst, "--version"]).startswith("uv"):     # a missing, empty or truncated file fails here
        return dst
    target, ext = _uv_target()
    url = "https://github.com/astral-sh/uv/releases/download/{}/uv-{}.{}".format(UV_VERSION, target, ext)
    try:
        data = _download(url, log)
        if hashlib.sha256(data).hexdigest() != UV_SHA256[target]:
            raise SetupError("checksum mismatch for " + url)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        name, part = os.path.basename(dst), dst + ".part"
        if ext == "zip":
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                member = next(m for m in z.namelist() if m.split("/")[-1] == name)
                with z.open(member) as src, open(part, "wb") as out:
                    shutil.copyfileobj(src, out)
        else:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
                member = next(m for m in t.getmembers() if m.name.split("/")[-1] == name)
                with t.extractfile(member) as src, open(part, "wb") as out:
                    shutil.copyfileobj(src, out)
            os.chmod(part, 0o700)           # only this user runs it
        os.replace(part, dst)
        return dst
    except Exception as e:
        found = shutil.which("uv")
        if found:
            log("Could not download uv ({}); using {}".format(e, found))
            return found
        raise SetupError("Could not download uv from GitHub: {}. Check the internet connection "
                         "(or install uv yourself and restart QGIS).".format(e))


def run(cmd, log, cancelled, env=None, cwd=None):
    """Run a command, stream its output to log, stop it when cancelled() turns true."""
    log("$ " + " ".join('"{}"'.format(c) if " " in c else c for c in cmd))
    # safe for bandit B603: cmd is built by this module (uv or the engine's python), a list without a shell
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env or clean_env(),  # nosec B603
                         cwd=cwd or home(), creationflags=NO_WINDOW, text=True, encoding="utf-8",
                         errors="replace", bufsize=1)
    tail = []

    def reader():
        for line in p.stdout:
            line = line.rstrip()
            if line:
                tail.append(line)
                del tail[:-30]
                log("  " + line)
    th = threading.Thread(target=reader, daemon=True)
    th.start()
    while p.poll() is None:
        if cancelled():
            p.kill()
            raise SetupError("Cancelled.")
        time.sleep(0.2)
    th.join(5)
    if p.returncode != 0:
        raise SetupError("Command failed (exit {}):\n{}".format(p.returncode, "\n".join(tail[-12:])))


def setup(log, progress, cancelled, model_key=None, force_cpu=False, with_model=False):
    """Create (or repair) the engine environment. Runs in a background task.

    The AI model is a separate download (download_model) unless with_model is set."""
    h = home()
    os.makedirs(h, exist_ok=True)
    info = detect()
    log("Computer: " + describe(info))
    for n in info["notes"]:
        log("Note: " + n)
    model_key = model_key or info["model"]
    force_cpu = force_cpu or EDITION == "cpu"
    backend = "cpu" if force_cpu and info["os"] != "Darwin" else info["torch_backend"]
    previous = load_state().get("torch_backend", "unset")
    # the new backend is saved only once its PyTorch is installed: a cancelled switch must reinstall on Repair
    st = {"env_version": ENV_VERSION, "complete": False, "detected": info, "model": model_key,
          "torch_backend": previous, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_state(st)
    progress(3)

    uv = get_uv(log)
    run([uv, "--version"], log, cancelled)
    progress(8)

    py = venv_python()
    if os.path.exists(py):
        try:
            ok = subprocess.run([py, "-c", "import sys; print(sys.version)"], capture_output=True,  # nosec B603
                                env=clean_env(), creationflags=NO_WINDOW, timeout=120).returncode == 0
        except subprocess.TimeoutExpired:
            raise SetupError("The engine's Python did not answer within 2 minutes. "
                             "Restart the computer and choose Settings… → Reinstall… in the Segmentot panel.")
        if not ok:
            log("Existing environment is broken, rebuilding it.")
            shutil.rmtree(os.path.join(h, "venv"), ignore_errors=True)
    if not os.path.exists(py):
        run([uv, "venv", "--python", PYTHON_VERSION, os.path.join(h, "venv")], log, cancelled)
    progress(15)

    cmd = [uv, "pip", "install", "--python", py, "torch", "torchvision"]
    if backend:
        cmd += ["--torch-backend", backend]
    if previous not in ("unset", backend):          # e.g. CPU -> CUDA after enabling the GPU
        cmd += ["--reinstall-package", "torch", "--reinstall-package", "torchvision"]
    log("Installing PyTorch (largest download: about 3 GB with CUDA, 300 MB–1 GB otherwise)…")
    try:
        run(cmd, log, cancelled)
    except SetupError as e:
        if backend in ("cpu", None) or "Cancelled" in str(e):
            raise
        log("GPU build of PyTorch could not be installed ({}). Falling back to the CPU build.".format(
            str(e).splitlines()[0]))
        backend = "cpu"
        run([uv, "pip", "install", "--python", py, "torch", "torchvision", "--torch-backend", "cpu",
             "--reinstall-package", "torch", "--reinstall-package", "torchvision"], log, cancelled)
    st["torch_backend"] = backend
    save_state(st)
    progress(65)
    run([uv, "pip", "install", "--python", py] + PACKAGES, log, cancelled)
    progress(75)

    try:
        torch_info = _probe_torch(py)
        gpu_error = torch_info.get("cuda_error")
    except SetupError as e:                          # the GPU build does not even start (CUDA libraries, hung driver)
        if backend in ("cpu", None):
            raise
        gpu_error = str(e).strip().splitlines()[-1][:200]
    if gpu_error:
        # e.g. "no kernel image is available": this PyTorch build does not support the card
        log("The GPU does not run with this PyTorch build ({}). Installing the CPU build instead.".format(gpu_error))
        run([uv, "pip", "install", "--python", py, "torch", "torchvision", "--torch-backend", "cpu",
             "--reinstall-package", "torch", "--reinstall-package", "torchvision"], log, cancelled)
        st["torch_backend"] = "cpu"
        save_state(st)
        torch_info = _probe_torch(py)
    elif backend not in ("cpu", None) and not torch_info["cuda_build"]:
        st["torch_backend"] = "cpu"                  # uv found no CUDA driver and chose the CPU build: retry on Repair
    if torch_info.get("mps_error"):                  # same wheel for Metal and CPU on macOS: the engine uses the CPU
        log("The Apple GPU (Metal) does not run here ({}); using the CPU.".format(torch_info["mps_error"]))
    log("PyTorch {} · CUDA build {} · GPU usable now: {}".format(
        torch_info["torch"], torch_info["cuda_build"] or "none",
        torch_info["gpu"] or ("Apple MPS" if torch_info["mps"] else "no (CPU)")))
    progress(80)

    if with_model:
        download_model(model_key, log, cancelled)
    progress(98)
    try:                                             # the cache is a second copy of every download
        run([uv, "cache", "clean"], log, cancelled)
    except SetupError as e:
        log("(could not clean the download cache: {})".format(e))
    st.update({"complete": True, "torch": torch_info, "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    save_state(st)
    progress(100)
    log("Done. The engine is installed." + ("" if model_downloaded(model_key) else
                                            " The AI model is not downloaded yet."))
    return st


PROBE = """
import json, torch
d = {"torch": torch.__version__, "cuda": torch.cuda.is_available(), "cuda_build": torch.version.cuda,
     "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()), "gpu": None}
for dev, flag in (("cuda", "cuda"), ("mps", "mps")):
    if d[flag]:
        try:      # run a real kernel: is_available() is true even when the build lacks this GPU's kernels
            x = torch.nn.functional.conv2d(torch.ones(1, 3, 8, 8, device=dev), torch.ones(4, 3, 3, 3, device=dev))
            float(x.sum())
            d["gpu"] = torch.cuda.get_device_name(0) if dev == "cuda" else "Apple GPU (MPS)"
        except Exception as e:
            d[flag] = False
            d[flag + "_error"] = str(e).splitlines()[0][:200]
print("PROBE" + json.dumps(d))
"""


def _probe_torch(py):
    try:
        out = subprocess.run([py, "-c", PROBE], capture_output=True, text=True,  # engine's python  # nosec B603
                             encoding="utf-8", errors="replace",
                             env=clean_env(), creationflags=NO_WINDOW, timeout=300)
    except subprocess.TimeoutExpired:
        raise SetupError("PyTorch did not start within 5 minutes (a hung GPU driver?). "
                         "Restart the computer and choose Settings… → Reinstall… in the Segmentot panel.")
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("PROBE")]
    if not line:
        raise SetupError("PyTorch does not start:\n" + (out.stderr or out.stdout)[-1500:])
    return json.loads(line[0][5:])


def model_downloaded(model_key):
    repo = os.path.join(home(), "hf", "hub", "models--" + MODELS[model_key][0].replace("/", "--"))
    try:                                             # offline loading uses the snapshot refs/main points at
        with open(os.path.join(repo, "refs", "main")) as f:
            commit = f.read().strip()
    except OSError:
        return False
    # huggingface_hub links a file into the snapshot only once it is complete; isfile() is False for a broken link
    return os.path.isfile(os.path.join(repo, "snapshots", commit, "model.safetensors"))


def delete_model(model_key):
    """Free the disk space of one downloaded model."""
    remove_repo(MODELS[model_key][0])


def remove_repo(model_id):
    """Delete a model's folder in the Hugging Face cache and its files in the shared blob store,
    except files another model also links to."""
    hub = os.path.realpath(os.path.join(home(), "hf", "hub"))
    repo = os.path.join(hub, "models--" + model_id.replace("/", "--"))

    def files(folder):          # the real files behind the links in a model folder
        return {os.path.realpath(os.path.join(d, f)) for d, _dirs, names in os.walk(folder) for f in names}
    mine = files(repo)
    for other in glob.glob(os.path.join(hub, "models--*")):
        if other != repo:
            mine -= files(other)
    shutil.rmtree(repo, ignore_errors=True)
    for f in mine:
        if f.startswith(hub + os.sep) and os.path.isfile(f):
            os.remove(f)


def download_model(model_key, log, cancelled):
    mid = MODELS[model_key][0]
    if "edgetam" in mid.lower():                # an engine installed before timm was in PACKAGES lacks it
        run([get_uv(log), "pip", "install", "--python", venv_python(), "timm"], log, cancelled)
    log("Downloading the model {} …".format(mid))
    # the engine's own loader: one place knows which classes load which model
    code = ("import sys; sys.path.insert(0, {!r}); import segmentot_server as s\n"
            "s.S['model_id'], s.S['device'] = {!r}, 'cpu'; s.load(); print('model ok')".format(
                os.path.dirname(SERVER), mid))
    run([venv_python(), "-c", code], log, cancelled)


# ------------------------------------------------------------------ offline engine package
def _platform_tag():
    return "{}-{}".format(platform.system().lower(), _uv_target()[0].split("-")[0])


def pack_name():
    t = load_state().get("torch") or {}
    kind = "" if platform.system() == "Darwin" else ("-gpu" if t.get("cuda_build") else "-cpu")
    return "segmentot-engine-{}{}.tar.gz".format(_platform_tag(), kind)


def _site_packages():
    out = subprocess.run([venv_python(), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],  # nosec B603
                         capture_output=True, text=True, encoding="utf-8", errors="replace", env=clean_env(),
                         creationflags=NO_WINDOW, timeout=120)
    if out.returncode != 0 or not out.stdout.strip():
        raise SetupError("The engine's Python does not start:\n" + (out.stderr or out.stdout)[-800:])
    return out.stdout.strip()


def export_pack(path, log, progress, cancelled, with_models=False):
    """Write the installed engine (its Python and packages, optionally the models) to one file that
    install_pack() installs on another computer of the same kind without internet."""
    if not is_ready():
        raise SetupError("Install the engine on this computer first.")
    h, st = home(), load_state()
    # a gated model may not be passed on: every computer has to ask for it itself
    models = [k for k in MODELS if model_downloaded(k) and k not in GATED] if with_models else []
    items = [(d, "python/" + os.path.basename(d)) for d in glob.glob(os.path.join(h, "python", "cpython-*"))]
    items.append((_site_packages(), "site-packages"))
    n_env = len(items)
    for k in models:
        repo = "models--" + MODELS[k][0].replace("/", "--")
        for sub in ("refs", "snapshots", ".no_exist"):       # not blobs/: the snapshot files are stored whole
            if os.path.isdir(os.path.join(h, "hf", "hub", repo, sub)):
                items.append((os.path.join(h, "hf", "hub", repo, sub), "hf/hub/{}/{}".format(repo, sub)))
    total = sum(len(files) for src, _ in items for _r, _d, files in os.walk(src)) or 1
    done = [0]

    def member(ti):
        if cancelled():
            raise SetupError("Cancelled.")
        if "__pycache__" in ti.name.split("/"):
            return None
        done[0] += ti.isfile()
        progress(2 + 96 * done[0] / total)
        return ti
    meta = json.dumps({"format": 1, "platform": _platform_tag(), "env_version": ENV_VERSION, "edition": EDITION,
                       "torch_backend": st.get("torch_backend"), "torch": st.get("torch"), "models": models,
                       "model": st.get("model"), "created": time.strftime("%Y-%m-%d %H:%M:%S")}).encode()
    log("Writing {} (this takes a few minutes)…".format(path))
    part = path + ".part"
    try:
        with tarfile.open(part, "w:gz", compresslevel=6) as tar:
            ti = tarfile.TarInfo("pack.json")           # first, so install_pack can check it before unpacking
            ti.size = len(meta)
            tar.addfile(ti, io.BytesIO(meta))
            for i, (src, name) in enumerate(items):
                log("  adding " + name)
                # model files are links into Hugging Face's blob store, which moves between versions
                tar.dereference = i >= n_env
                tar.add(src, name, filter=member)
        os.replace(part, path)
    finally:
        if os.path.exists(part):
            os.remove(part)
    progress(100)
    log("Done: {:.0f} MB. Models inside: {}.".format(os.path.getsize(path) / 1e6,
                                                     ", ".join(MODELS[k][1] for k in models) or "none"))


def install_pack(path, log, progress, cancelled):
    """Install the engine from a package made by export_pack(). No internet needed."""
    h = home()
    os.makedirs(h, exist_ok=True)
    tmp = os.path.join(h, "pack.tmp")
    raw = None
    try:
        raw = open(path, "rb")
        tar = tarfile.open(fileobj=raw, mode="r:gz")
        first = tar.next()
        meta = json.load(tar.extractfile(first)) if first is not None and first.name == "pack.json" else None
    except (OSError, ValueError, tarfile.TarError, EOFError) as e:
        if raw:
            raw.close()
        raise SetupError("Cannot read {}: {}".format(path, e))
    with raw, tar:
        if not meta or meta.get("format") != 1:
            raise SetupError("{} is not a Segmentot engine package.".format(os.path.basename(path)))
        if meta.get("platform") != _platform_tag():
            raise SetupError("This package is for another kind of computer ({}); this one is {}.".format(
                meta.get("platform"), _platform_tag()))
        if meta.get("env_version") != ENV_VERSION:
            raise SetupError("This package was made by another version of the plugin. Make a new one, "
                             "or use the online install.")
        log("Installing the engine from {} (no download)…".format(path))
        stop_engine()
        old = load_state()
        st = {"env_version": ENV_VERSION, "complete": False, "torch_backend": meta.get("torch_backend"),
              "model": old.get("model") or meta.get("model"), "pack": os.path.basename(path),
              "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        save_state(st)
        shutil.rmtree(tmp, ignore_errors=True)
        size = os.path.getsize(path) or 1
        kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        for m in tar:
            if cancelled():
                raise SetupError("Cancelled.")
            if os.path.isabs(m.name) or ".." in m.name.replace("\\", "/").split("/"):
                raise SetupError("Unsafe path in the package: " + m.name)
            try:
                tar.extract(m, tmp, **kw)
            except tarfile.TarError as e:
                raise SetupError("Cannot unpack {}: {}".format(m.name, e))
            progress(2 + 83 * raw.tell() / size)
    for name in ("python", "venv"):
        shutil.rmtree(os.path.join(h, name), ignore_errors=True)
    os.replace(os.path.join(tmp, "python"), os.path.join(h, "python"))
    base = glob.glob(os.path.join(h, "python", "cpython-*", "python.exe" if WIN else os.path.join("bin", "python3")))
    if not base:
        raise SetupError("The package has no Python in it.")
    # a fresh venv: the packaged one would point at the folder of the computer that made it
    run([base[0], "-m", "venv", "--without-pip", os.path.join(h, "venv")], log, cancelled)
    progress(90)
    site = _site_packages()
    shutil.rmtree(site, ignore_errors=True)
    os.replace(os.path.join(tmp, "site-packages"), site)
    if os.path.isdir(os.path.join(tmp, "hf")):
        shutil.copytree(os.path.join(tmp, "hf"), os.path.join(h, "hf"), symlinks=True, dirs_exist_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)
    torch_info = _probe_torch(venv_python())
    log("PyTorch {} · CUDA build {} · GPU usable now: {}".format(
        torch_info["torch"], torch_info["cuda_build"] or "none",
        torch_info["gpu"] or ("Apple MPS" if torch_info["mps"] else "no (CPU)")))
    st.update({"complete": True, "detected": detect(), "torch": torch_info,
               "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    st["model"] = st["model"] or st["detected"]["model"]
    save_state(st)
    progress(100)
    log("Done. The engine is installed." + ("" if model_downloaded(st["model"]) else
                                            " The AI model is not downloaded yet."))
    return st


def remove_all():
    """Delete only what setup() created: SEGMENTOT_HOME may be a folder that also holds the user's own files."""
    stop_engine()
    h = home()
    for p in (uv_path(), uv_path() + ".part", os.path.join(h, "env.json"), os.path.join(h, "engine.log"),
              os.path.join(h, "engine.key")):
        try:
            os.remove(p)
        except OSError:
            pass
    for name in ("python", "venv", "hf", "cache", "pack.tmp"):
        shutil.rmtree(os.path.join(h, name), ignore_errors=True)
    for d in (os.path.join(h, "bin"), h):           # only when empty
        try:
            os.rmdir(d)
        except OSError:
            pass


# ------------------------------------------------------------------ engine process
_proc = None
_proc_lock = threading.Lock()


def engine_running_here():
    return _proc is not None and _proc.poll() is None


def start_engine(port, model_key):
    """Start our engine process (no-op if it is already running)."""
    global _proc
    with _proc_lock:
        if engine_running_here():
            return
        if not is_ready():
            raise SetupError("not installed")
        key = engine_key(create=True)
        if not key:
            raise SetupError("Cannot write the engine key in {}.".format(home()))
        cmd = [venv_python(), "-u", SERVER, "--port", str(port), "--model", MODELS[model_key][0],
               "--parent-pid", str(os.getpid()), "--idle-exit", "1800"]
        if EDITION == "cpu":
            cmd += ["--device", "cpu"]
        logf = open(os.path.join(home(), "engine.log"), "a", encoding="utf-8")
        logf.write("\n==== {} start {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S"), " ".join(cmd)))
        logf.flush()
        kw = {"creationflags": NO_WINDOW} if WIN else {"start_new_session": True}
        # safe for bandit B603: the engine's own python and server script, a list without a shell
        _proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,  # nosec B603
                                 # offline: models come from download_model only. The key goes by the
                                 # environment, not the command line: other users can read command lines
                                 env=clean_env({"HF_HUB_OFFLINE": "1", "SEGMENTOT_KEY": key}),
                                 cwd=home(), **kw)
        logf.close()


def stop_engine():
    global _proc
    with _proc_lock:
        if engine_running_here():
            _proc.terminate()
            try:
                _proc.wait(5)
            except subprocess.TimeoutExpired:
                _proc.kill()
        _proc = None


def engine_log_tail(n=15):
    try:
        with open(os.path.join(home(), "engine.log"), encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""
