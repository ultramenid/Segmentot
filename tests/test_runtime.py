"""Tests for runtime.py: plain `python3 tests/test_runtime.py`, no QGIS, no network, nothing installed.

runtime.py uses a relative import, so it is imported as a package through a symlink named Segmentot
in a temporary folder (or set SEGMENTOT_TEST_PARENT to a folder that holds the package as Segmentot/).
"""
import atexit
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.environ.get("SEGMENTOT_TEST_PARENT")
if not PARENT:
    PARENT = tempfile.mkdtemp()
    os.symlink(REPO, os.path.join(PARENT, "Segmentot"))
    atexit.register(shutil.rmtree, PARENT, True)
sys.path.insert(0, PARENT)
sys.dont_write_bytecode = True
from Segmentot.core import runtime  # noqa: E402


class HomeTest(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp()
        p = mock.patch.dict(os.environ, {"SEGMENTOT_HOME": self.home})
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(shutil.rmtree, self.home, True)

    def touch(self, *parts, data=b"x"):
        p = os.path.join(self.home, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)
        return p


class RemoveAll(HomeTest):
    def make_engine(self):
        uv = os.path.basename(runtime.uv_path())
        for parts in (("bin", uv), ("bin", uv + ".part"),
                      ("python", "cpython-3.12", "lib.so"), ("venv", "pyvenv.cfg"), ("hf", "hub", "x"),
                      ("cache", "wheel"), ("env.json",), ("engine.log",), ("engine.key",)):
            self.touch(*parts)

    def test_keeps_unrelated_files(self):
        self.make_engine()
        mine = self.touch("my_notes.txt")
        theirs = self.touch("Projects", "data.tif")
        runtime.remove_all()
        self.assertEqual(sorted(os.listdir(self.home)), ["Projects", "my_notes.txt"])
        self.assertTrue(os.path.isfile(mine) and os.path.isfile(theirs))

    def test_removes_empty_home(self):
        self.make_engine()
        runtime.remove_all()
        self.assertFalse(os.path.exists(self.home))


class EngineKey(HomeTest):
    def test_made_once_and_private(self):
        self.assertIsNone(runtime.engine_key())                 # looking never makes one
        self.assertEqual(os.listdir(self.home), [])
        key = runtime.engine_key(create=True)
        self.assertEqual(len(key), 64)
        self.assertEqual(runtime.engine_key(), key)
        self.assertEqual(runtime.engine_key(create=True), key)
        if os.name != "nt":
            self.assertEqual(os.stat(os.path.join(self.home, "engine.key")).st_mode & 0o777, 0o600)

    def test_empty_file_is_replaced(self):
        self.touch("engine.key", data=b"")
        self.assertIsNone(runtime.engine_key())
        self.assertEqual(len(runtime.engine_key(create=True)), 64)


class ModelDownloaded(HomeTest):
    def repo(self, *parts):
        return os.path.join(self.home, "hf", "hub", "models--facebook--sam2.1-hiera-tiny", *parts)

    def snapshot(self, commit="abc123"):
        self.touch("hf", "hub", "models--facebook--sam2.1-hiera-tiny", "refs", "main", data=commit.encode())
        snap = self.repo("snapshots", commit)
        os.makedirs(snap)
        return snap

    def test_missing_repo(self):
        self.assertFalse(runtime.model_downloaded("tiny"))

    def test_empty_snapshot(self):
        self.snapshot()
        self.assertFalse(runtime.model_downloaded("tiny"))

    def test_config_only(self):
        with open(os.path.join(self.snapshot(), "config.json"), "w") as f:
            f.write("{}")
        self.assertFalse(runtime.model_downloaded("tiny"))

    @unittest.skipIf(os.name == "nt", "symlinks need privileges on Windows")
    def test_broken_symlink(self):
        os.symlink(self.repo("blobs", "deadbeef"), os.path.join(self.snapshot(), "model.safetensors"))
        self.assertFalse(runtime.model_downloaded("tiny"))

    @unittest.skipIf(os.name == "nt", "symlinks need privileges on Windows")
    def test_symlink_to_blob(self):
        blob = self.touch("hf", "hub", "models--facebook--sam2.1-hiera-tiny", "blobs", "deadbeef")
        os.symlink(blob, os.path.join(self.snapshot(), "model.safetensors"))
        self.assertTrue(runtime.model_downloaded("tiny"))
        self.assertFalse(runtime.model_downloaded("small"))

    def test_real_file(self):
        with open(os.path.join(self.snapshot(), "model.safetensors"), "wb") as f:
            f.write(b"weights")
        self.assertTrue(runtime.model_downloaded("tiny"))

    def test_new_revision_not_finished(self):
        with open(os.path.join(self.snapshot("old"), "model.safetensors"), "wb") as f:
            f.write(b"weights")
        self.snapshot("new")                         # refs/main now points at an unfinished download
        self.assertFalse(runtime.model_downloaded("tiny"))


class GetUv(HomeTest):
    BINARY = b"\x7fELF pretend uv binary"

    def archive(self):
        target, ext = runtime._uv_target()
        name = os.path.basename(runtime.uv_path())
        buf = io.BytesIO()
        if ext == "zip":
            with zipfile.ZipFile(buf, "w") as z:
                z.writestr(name, self.BINARY)
        else:
            with tarfile.open(fileobj=buf, mode="w:gz") as t:
                info = tarfile.TarInfo("uv-{}/{}".format(target, name))
                info.size = len(self.BINARY)
                t.addfile(info, io.BytesIO(self.BINARY))
        return target, buf.getvalue()

    def get(self, data, sha=None):
        urls = []

        def fake_download(url, log, timeout=120):
            urls.append(url)
            return data
        patches = [mock.patch.object(runtime, "_download", fake_download),
                   mock.patch("shutil.which", return_value=None)]
        if sha:
            patches.append(mock.patch.dict(runtime.UV_SHA256, sha))
        for p in patches:
            p.start()
        try:
            return runtime.get_uv(lambda s: None)
        finally:
            for p in patches:
                p.stop()
            if urls:
                self.assertIn("/download/{}/".format(runtime.UV_VERSION), urls[0])
                self.assertNotIn("latest", urls[0])

    def test_rejects_wrong_hash(self):
        _, data = self.archive()
        with self.assertRaises(runtime.SetupError) as cm:
            self.get(data)
        self.assertIn("checksum", str(cm.exception))
        self.assertFalse(os.path.exists(runtime.uv_path()))
        self.assertFalse(os.path.exists(runtime.uv_path() + ".part"))

    def test_accepts_matching_hash(self):
        target, data = self.archive()
        dst = self.get(data, {target: hashlib.sha256(data).hexdigest()})
        self.assertEqual(dst, runtime.uv_path())
        with open(dst, "rb") as f:
            self.assertEqual(f.read(), self.BINARY)
        self.assertFalse(os.path.exists(dst + ".part"))
        if os.name != "nt":
            self.assertTrue(os.access(dst, os.X_OK))

    def test_replaces_empty_binary(self):
        dst = self.touch("bin", os.path.basename(runtime.uv_path()), data=b"")
        os.chmod(dst, 0o755)
        target, data = self.archive()
        self.assertEqual(self.get(data, {target: hashlib.sha256(data).hexdigest()}), dst)
        with open(dst, "rb") as f:
            self.assertEqual(f.read(), self.BINARY)

    @unittest.skipIf(os.name == "nt", "needs a shell script")
    def test_reuses_working_binary(self):
        dst = self.touch("bin", "uv", data=b"#!/bin/sh\necho 'uv 0.13.7'\n")
        os.chmod(dst, 0o755)
        with mock.patch.object(runtime, "_download", side_effect=AssertionError("downloaded")):
            self.assertEqual(runtime.get_uv(lambda s: None), dst)

    def test_pinned_hashes_cover_every_target(self):
        for system in ("Windows", "Darwin", "Linux"):
            for machine in ("x86_64", "AMD64", "arm64", "aarch64"):
                with mock.patch("platform.system", return_value=system), \
                        mock.patch("platform.machine", return_value=machine):
                    self.assertRegex(runtime.UV_SHA256[runtime._uv_target()[0]], "^[0-9a-f]{64}$")


class Detect(unittest.TestCase):
    def test_memory_not_available(self):
        def fake_run(cmd, timeout=15):
            return "8.6\n" if "--query-gpu=compute_cap" in cmd else "NVIDIA RTX A2000, [N/A], 550.54\n"
        with mock.patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
                mock.patch.object(runtime, "_run_quiet", fake_run), \
                mock.patch.object(runtime, "_libcuda_loads", return_value=True):
            info = runtime.detect()
        self.assertIsNone(info["nvidia"]["vram_gb"])
        self.assertEqual(info["nvidia"]["driver"], "550.54")
        self.assertIn("NVIDIA RTX A2000", runtime.describe(dict(info, accel="cuda")))


GPU_INFO = {"os": "Linux", "arch": "x86_64", "flatpak": False, "accel": "cuda", "torch_backend": "auto",
            "nvidia": {"name": "NVIDIA RTX 3060", "vram_gb": 12.0, "driver": "550.54"},
            "model": "base", "notes": [], "edition": "gpu", "gpu_missing": False}
MAC_INFO = dict(GPU_INFO, os="Darwin", arch="arm64", accel="mps", torch_backend=None, nvidia=None)
CUDA_TORCH = {"torch": "2.9.0", "cuda": True, "cuda_build": "12.8", "mps": False, "gpu": "NVIDIA RTX 3060"}
CPU_TORCH = {"torch": "2.9.0", "cuda": False, "cuda_build": None, "mps": False, "gpu": None}


class Setup(HomeTest):
    def setup_with(self, info=GPU_INFO, probes=(CUDA_TORCH,), fail=None):
        """Run setup() with every command faked; fail(cmd) may raise. Returns the commands run."""
        cmds, probes = [], list(probes)

        def fake_run(cmd, log, cancelled, env=None, cwd=None):
            cmds.append(cmd)
            if fail:
                fail(cmd)

        def fake_probe(py):
            r = probes.pop(0)
            if isinstance(r, Exception):
                raise r
            return r
        with mock.patch.object(runtime, "EDITION", "gpu"), \
                mock.patch.object(runtime, "detect", return_value=dict(info)), \
                mock.patch.object(runtime, "get_uv", return_value="uv"), \
                mock.patch.object(runtime, "run", fake_run), \
                mock.patch.object(runtime, "_probe_torch", fake_probe), \
                mock.patch.object(runtime, "download_model"):
            try:
                runtime.setup(lambda s: None, lambda p: None, lambda: False)
            finally:
                self.cmds = cmds
        return cmds

    @staticmethod
    def torch_cmds(cmds):
        return [c for c in cmds if "torch" in c]

    def test_cancelled_cpu_to_gpu_switch_reinstalls_on_repair(self):
        runtime.save_state({"torch_backend": "cpu", "complete": True})

        def cancel_torch(cmd):
            if "torch" in cmd:
                raise runtime.SetupError("Cancelled.")
        with self.assertRaises(runtime.SetupError):
            self.setup_with(fail=cancel_torch)
        self.assertEqual(runtime.load_state()["torch_backend"], "cpu")
        self.assertFalse(runtime.load_state()["complete"])

        cmds = self.setup_with()                     # Repair
        torch = self.torch_cmds(cmds)[0]
        self.assertIn("--reinstall-package", torch)
        self.assertEqual(torch[torch.index("--torch-backend") + 1], "auto")
        st = runtime.load_state()
        self.assertEqual(st["torch_backend"], "auto")
        self.assertTrue(st["complete"])

    def test_fresh_install_cancelled_keeps_no_backend(self):
        def cancel_torch(cmd):
            if "torch" in cmd:
                raise runtime.SetupError("Cancelled.")
        with self.assertRaises(runtime.SetupError):
            self.setup_with(fail=cancel_torch)
        self.assertEqual(runtime.load_state().get("torch_backend", "unset"), "unset")

    def test_gpu_install_fails_falls_back_to_cpu(self):
        def gpu_fails(cmd):
            if "torch" in cmd and cmd[cmd.index("--torch-backend") + 1] == "auto":
                raise runtime.SetupError("Command failed (exit 1):\nno space")
        self.setup_with(probes=(CPU_TORCH,), fail=gpu_fails)
        self.assertEqual([c[c.index("--torch-backend") + 1] for c in self.torch_cmds(self.cmds)], ["auto", "cpu"])
        self.assertEqual(runtime.load_state()["torch_backend"], "cpu")

    def test_cuda_error_falls_back_to_cpu(self):
        bad = dict(CUDA_TORCH, cuda=False, gpu=None, cuda_error="no kernel image is available")
        self.setup_with(probes=(bad, CPU_TORCH))
        self.assertEqual([c[c.index("--torch-backend") + 1] for c in self.torch_cmds(self.cmds)], ["auto", "cpu"])
        self.assertEqual(runtime.load_state()["torch_backend"], "cpu")

    def test_probe_crash_on_gpu_build_falls_back_to_cpu(self):
        crash = runtime.SetupError("PyTorch does not start:\nOSError: [WinError 126] cudnn64_9.dll")
        self.setup_with(probes=(crash, CPU_TORCH))
        self.assertEqual([c[c.index("--torch-backend") + 1] for c in self.torch_cmds(self.cmds)], ["auto", "cpu"])
        st = runtime.load_state()
        self.assertEqual(st["torch_backend"], "cpu")
        self.assertTrue(st["complete"])

    def test_probe_crash_on_cpu_build_is_an_error(self):
        crash = runtime.SetupError("PyTorch does not start:\nImportError: numpy")
        with self.assertRaises(runtime.SetupError):
            self.setup_with(probes=(crash, crash), fail=lambda cmd: None, info=dict(GPU_INFO, torch_backend="cpu"))
        self.assertEqual(len(self.torch_cmds(self.cmds)), 1)

    def test_uv_chose_cpu_build_retries_gpu_on_repair(self):
        self.setup_with(probes=(CPU_TORCH,))
        self.assertEqual(runtime.load_state()["torch_backend"], "cpu")
        self.setup_with()
        self.assertIn("--reinstall-package", self.torch_cmds(self.cmds)[0])

    def test_mps_error_keeps_the_install(self):
        self.setup_with(info=MAC_INFO, probes=(dict(CPU_TORCH, mps_error="MPS backend out of memory"),))
        self.assertEqual(len(self.torch_cmds(self.cmds)), 1)
        self.assertNotIn("--torch-backend", self.torch_cmds(self.cmds)[0])
        self.assertTrue(runtime.load_state()["complete"])


class Models(HomeTest):
    def test_every_model_has_a_documentation_page_and_a_size(self):
        for key, (mid, label, mb) in runtime.MODELS.items():
            self.assertEqual(runtime.model_docs(key), "https://huggingface.co/" + mid)
            self.assertTrue(label and mb > 0)
        self.assertEqual((runtime.size_text(323), runtime.size_text(3440)), ("323 MB", "3.4 GB"))
        self.assertTrue(set(runtime.GATED) <= set(runtime.MODELS))

    def download(self, key):
        with mock.patch.object(runtime, "get_uv", return_value="uv"), mock.patch.object(runtime, "run") as run:
            runtime.download_model(key, lambda s: None, lambda: False)
        return [c.args[0] for c in run.call_args_list]

    def test_edgetam_gets_its_extra_package_before_the_download(self):
        cmds = self.download("edge")
        self.assertEqual(cmds[0][:3] + cmds[0][-1:], ["uv", "pip", "install", "timm"])
        self.assertIn("'yonigozlan/EdgeTAM-hf'", cmds[1][-1])

    def test_download_uses_the_engine_loader_and_installs_nothing_for_sam2(self):
        cmds = self.download("small")
        self.assertEqual(len(cmds), 1)
        self.assertIn("segmentot_server", cmds[0][-1])
        self.assertIn("'facebook/sam2.1-hiera-small'", cmds[0][-1])

    @unittest.skipIf(os.name == "nt", "file modes are a POSIX thing")
    def test_token_is_saved_where_huggingface_looks_and_only_this_user_can_read_it(self):
        runtime.save_hf_token("  hf_abc\n")
        path = os.path.join(self.home, "hf", "token")
        self.assertEqual(runtime.token_path(), path)
        with open(path) as f:
            self.assertEqual(f.read(), "hf_abc")
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)

    def model(self, mid, commit, own, shared):
        """A downloaded model in huggingface_hub's layout: snapshot links -> files in the shared blob store."""
        repo = "models--" + mid.replace("/", "--")
        self.touch("hf", "hub", repo, "refs", "main", data=commit.encode())
        snap = os.path.join(self.home, "hf", "hub", repo, "snapshots", commit)
        os.makedirs(snap)
        os.symlink(own, os.path.join(snap, "model.safetensors"))
        os.symlink(shared, os.path.join(snap, "preprocessor_config.json"))

    @unittest.skipIf(os.name == "nt", "symlinks need privileges on Windows")
    def test_delete_frees_the_models_own_files_and_keeps_shared_ones(self):
        tiny = self.touch("hf", "hub", "blobs", "aa", "tiny-weights")
        small = self.touch("hf", "hub", "blobs", "bb", "small-weights")
        shared = self.touch("hf", "hub", "blobs", "cc", "same-config")
        keep = self.touch("my-notes.txt")
        self.model(runtime.MODELS["tiny"][0], "c1", tiny, shared)
        self.model(runtime.MODELS["small"][0], "c2", small, shared)
        runtime.delete_model("tiny")
        self.assertFalse(runtime.model_downloaded("tiny"))
        self.assertFalse(os.path.exists(tiny))
        self.assertTrue(runtime.model_downloaded("small"))
        self.assertTrue(os.path.exists(small) and os.path.exists(shared) and os.path.exists(keep))
        runtime.delete_model("small")
        self.assertFalse(os.path.exists(small) or os.path.exists(shared))
        runtime.delete_model("large")               # never downloaded: nothing to do, no error


class ProbeTorch(HomeTest):
    def test_timeout_is_a_setup_error(self):
        with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired("python", 300)):
            with self.assertRaises(runtime.SetupError) as cm:
                runtime._probe_torch("python")
        self.assertIn("did not start", str(cm.exception))

    def test_non_utf8_output(self):
        script = ("import sys\nsys.stderr.buffer.write(b'warning C:\\\\Users\\\\J\\xfcrgen\\xff\\n')\n"
                  "sys.stderr.flush()\nprint('PROBE' + '{\"torch\": \"x\"}')\n")
        with mock.patch.object(runtime, "PROBE", script):
            self.assertEqual(runtime._probe_torch(sys.executable), {"torch": "x"})


class ModelIsSeparate(HomeTest):
    def run_setup(self, **kw):
        with mock.patch.object(runtime, "EDITION", "gpu"), \
                mock.patch.object(runtime, "detect", return_value=dict(GPU_INFO)), \
                mock.patch.object(runtime, "get_uv", return_value="uv"), \
                mock.patch.object(runtime, "run"), \
                mock.patch.object(runtime, "_probe_torch", return_value=CUDA_TORCH), \
                mock.patch.object(runtime, "download_model") as download:
            runtime.setup(lambda s: None, lambda p: None, lambda: False, **kw)
        return download

    def test_install_does_not_download_the_model_by_default(self):
        self.assertFalse(self.run_setup().called)
        self.assertTrue(runtime.load_state()["complete"])

    def test_install_downloads_the_model_when_asked(self):
        self.assertTrue(self.run_setup(with_model=True).called)

    def test_engine_never_downloads_a_model_itself(self):
        with mock.patch.object(runtime, "is_ready", return_value=True), \
                mock.patch.object(runtime.subprocess, "Popen") as popen:
            runtime.start_engine(8776, "base")
        self.addCleanup(setattr, runtime, "_proc", None)
        self.assertEqual(popen.call_args.kwargs["env"]["HF_HUB_OFFLINE"], "1")

    def test_engine_gets_the_key_by_the_environment_only(self):
        with mock.patch.object(runtime, "is_ready", return_value=True), \
                mock.patch.object(runtime.subprocess, "Popen") as popen:
            runtime.start_engine(8776, "base")
        self.addCleanup(setattr, runtime, "_proc", None)
        key = runtime.engine_key()
        self.assertEqual(popen.call_args.kwargs["env"]["SEGMENTOT_KEY"], key)
        self.assertNotIn(key, " ".join(popen.call_args.args[0]))
        with open(os.path.join(self.home, "engine.log")) as f:
            self.assertNotIn(key, f.read())


class Pack(HomeTest):
    quiet = (lambda s: None, lambda p: None, lambda: False)

    def tar(self, members, meta=None):
        """A .tar.gz with pack.json first (unless meta is False) and the given (name, bytes) members."""
        path = os.path.join(tempfile.mkdtemp(dir=self.home), "pack.tar.gz")
        if meta is not False:
            import json
            data = json.dumps(dict({"format": 1, "platform": runtime._platform_tag(),
                                    "env_version": runtime.ENV_VERSION}, **(meta or {}))).encode()
            members = [("pack.json", data)] + list(members)
        with tarfile.open(path, "w:gz") as t:
            for name, data in members:
                ti = tarfile.TarInfo(name)
                ti.size = len(data)
                t.addfile(ti, io.BytesIO(data))
        return path

    def test_rejects_a_file_that_is_not_a_package(self):
        junk = self.touch("junk.tar.gz", data=b"not a tar at all")
        for path in (junk, self.tar([("python/x", b"1")], meta=False)):
            with self.assertRaises(runtime.SetupError):
                runtime.install_pack(path, *self.quiet)
        self.assertFalse(os.path.exists(os.path.join(self.home, "python")))

    def test_rejects_a_package_for_another_kind_of_computer(self):
        with self.assertRaises(runtime.SetupError) as e:
            runtime.install_pack(self.tar([], meta={"platform": "plan9-mips"}), *self.quiet)
        self.assertIn("plan9-mips", str(e.exception))

    def test_rejects_paths_that_leave_the_engine_folder(self):
        outside = os.path.join(os.path.dirname(self.home), "segmentot_evil_test")
        with self.assertRaises(runtime.SetupError):
            runtime.install_pack(self.tar([("../../segmentot_evil_test", b"x")]), *self.quiet)
        self.assertFalse(os.path.exists(outside))
        self.assertFalse(runtime.is_ready())

    @unittest.skipIf(runtime.WIN, "uses a shell-script stand-in for the packaged Python")
    def test_export_then_install_on_another_computer(self):
        # "computer A": a small real environment standing in for the installed engine
        py = self.touch("python", "cpython-test", "bin", "python3",
                        data='#!/bin/sh\nexec "{}" "$@"\n'.format(sys.executable).encode())
        os.chmod(py, 0o755)
        subprocess.run([py, "-m", "venv", "--without-pip", os.path.join(self.home, "venv")], check=True)
        runtime.save_state({"env_version": runtime.ENV_VERSION, "complete": True, "model": "base",
                            "torch_backend": "cpu", "torch": CPU_TORCH})
        site = runtime._site_packages()
        os.makedirs(os.path.join(site, "hello_pack", "__pycache__"))
        for name, data in (("__init__.py", b"VALUE = 42\n"), (os.path.join("__pycache__", "junk.pyc"), b"x")):
            with open(os.path.join(site, "hello_pack", name), "wb") as f:
                f.write(data)
        # the model as huggingface_hub stores it: the snapshot links to a blob store outside the repo folder
        repo = ("hf", "hub", "models--facebook--sam2.1-hiera-base-plus")
        self.touch(*repo, "refs", "main", data=b"abc")
        blob = self.touch("hf", "hub", "blobs", "84", "weights", data=b"WEIGHTS")
        os.makedirs(os.path.join(self.home, *repo, "snapshots", "abc"))
        os.symlink(blob, os.path.join(self.home, *repo, "snapshots", "abc", "model.safetensors"))
        pack = os.path.join(tempfile.mkdtemp(dir=self.home), runtime.pack_name())
        runtime.export_pack(pack, *self.quiet, with_models=True)
        with tarfile.open(pack) as t:
            names = t.getnames()
        self.assertEqual(names[0], "pack.json")
        self.assertFalse([n for n in names if "__pycache__" in n])

        # "computer B": an empty engine folder, same kind of computer
        other = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, other, True)
        with mock.patch.dict(os.environ, {"SEGMENTOT_HOME": other}), \
                mock.patch.object(runtime, "_probe_torch", return_value=CPU_TORCH), \
                mock.patch.object(runtime, "detect", return_value=dict(GPU_INFO, model="small")):
            self.assertFalse(runtime.is_ready())
            runtime.install_pack(pack, *self.quiet)
            self.assertTrue(runtime.is_ready())
            self.assertTrue(runtime.model_downloaded("base"))            # the model travelled in the package
            weights = os.path.join(other, *repo, "snapshots", "abc", "model.safetensors")
            self.assertFalse(os.path.islink(weights))
            with open(weights, "rb") as f:
                self.assertEqual(f.read(), b"WEIGHTS")
            self.assertFalse(runtime.model_downloaded("tiny"))
            self.assertEqual(runtime.load_state()["model"], "base")
            out = subprocess.run([runtime.venv_python(), "-c", "import hello_pack; print(hello_pack.VALUE)"],
                                 capture_output=True, text=True, env=runtime.clean_env())
            self.assertEqual(out.stdout.strip(), "42", out.stderr)
            self.assertFalse(os.path.exists(os.path.join(other, "pack.tmp")))


class Permissions(HomeTest):
    @unittest.skipIf(runtime.WIN or os.geteuid() == 0, "needs POSIX permissions and a normal user")
    def test_home_writable(self):
        self.assertTrue(runtime.home_writable())
        locked = tempfile.mkdtemp()
        os.chmod(locked, 0o555)
        self.addCleanup(shutil.rmtree, locked, True)
        self.addCleanup(os.chmod, locked, 0o755)
        with mock.patch.dict(os.environ, {"SEGMENTOT_HOME": os.path.join(locked, "engine")}):
            self.assertFalse(runtime.home_writable())

    def test_guide_for_mac_and_linux_names_the_folder_and_the_user(self):
        with mock.patch.object(runtime, "WIN", False), \
                mock.patch.object(runtime.getpass, "getuser", return_value="ani"):
            title, how, cmd = runtime.permission_guide()[0]
        self.assertEqual(cmd, 'sudo mkdir -p "{0}" && sudo chown -R "ani" "{0}"'.format(self.home))
        self.assertIn("Terminal", how)

    def test_guide_for_windows_is_powershell_for_the_logged_in_account(self):
        with mock.patch.object(runtime, "WIN", True), \
                mock.patch.dict(os.environ, {"USERDOMAIN": "KANTOR", "USERNAME": "ani"}):
            guide = runtime.permission_guide()
        self.assertEqual(guide[0][2], 'New-Item -ItemType Directory -Force "{0}" | Out-Null\n'
                                      'icacls "{0}" /grant "KANTOR\\ani:(OI)(CI)F" /T'.format(self.home))
        self.assertIn("Run as administrator", guide[0][1])
        self.assertIn("HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem", guide[1][2])
        self.assertIn("LongPathsEnabled", guide[1][2])


if __name__ == "__main__":
    unittest.main()
