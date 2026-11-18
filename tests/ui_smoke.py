"""Drive the setup UI headless: the two setup steps, startup check, permission guide, install-from-file
through the panel, and the package export. Needs an engine package file as its argument."""
import atexit
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.environ.get("SEGMENTOT_TEST_PARENT") or os.path.dirname(REPO))
sys.path.insert(0, HERE)
if "SEGMENTOT_HOME" not in os.environ:  # never install over the real engine folder
    os.environ["SEGMENTOT_HOME"] = tempfile.mkdtemp(prefix="segmentot_ui_")
    atexit.register(shutil.rmtree, os.environ["SEGMENTOT_HOME"], True)
import e2e  # noqa: E402  (the imports below need the paths set above)
from qgis.core import QgsApplication  # noqa: E402
from qgis.gui import QgsMapCanvas  # noqa: E402
from qgis.PyQt.QtCore import QSettings, QTimer  # noqa: E402
from qgis.PyQt.QtWidgets import QApplication, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton  # noqa: E402
from Segmentot.core import runtime  # noqa: E402
from Segmentot.gui import setup_ui, plugin as plugin_mod  # noqa: E402

assert len(sys.argv) > 1, "usage: ui_smoke.py <engine package .tar.gz>  (make one in Settings → Save engine package…)"
PACK = sys.argv[1]
app = QgsApplication([], True)
app.setOrganizationName("SegmentotE2E")
app.setApplicationName("SegmentotUI")
app.initQgis()
R = e2e.Run()
win = QMainWindow()
canvas = QgsMapCanvas()
win.setCentralWidget(canvas)
win.resize(1200, 800)
win.show()


def pump(pred, timeout, what):
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents()
        if pred():
            return time.time() - t0
        time.sleep(0.02)
    raise AssertionError("timed out waiting for " + what)


def press(label):
    """While a modal box is open: note what it shows, then click the button whose text starts with label."""
    seen = {}

    def go():
        box = QApplication.activeModalWidget()
        if box is None:
            return QTimer.singleShot(50, go)
        seen["text"] = box.text() if isinstance(box, QMessageBox) else box.windowTitle()
        seen["commands"] = [e.toPlainText() for e in box.findChildren(QPlainTextEdit)]
        next(b for b in box.findChildren(QPushButton) if b.text().replace("&", "").startswith(label)).click()

    QTimer.singleShot(50, go)
    return seen


# 1. a fresh computer: the panel shows the two setup steps (fresh engine folder, set by SEGMENTOT_HOME)
QSettings().setValue("Segmentot/port", 8791)  # not the port of a QGIS that may be open on this computer
QSettings().remove("Segmentot/model")
iface = e2e.FakeIface(win, canvas)
plugin = plugin_mod.SegmentotPlugin(iface)
plugin.initGui()
plugin.dock.show()
app.processEvents()
R.check(
    "setup: a fresh computer sees the setup steps, not the work screen",
    not runtime.is_ready() and not plugin.setup_page.isHidden() and plugin.work_page.isHidden(),
)
R.check(
    "setup: step 1 says where it will run, the download size and the free disk space",
    plugin.step1.title() == "Step 1 · Install the engine"
    and "Will run on" in plugin.step1_text.text()
    and "GB" in plugin.step1_text.text()
    and "free disk space" in plugin.step1_text.text()
    and not plugin.install_btn.isHidden()
    and not plugin.file_btn.isHidden(),
    plugin.step1_text.text()[-75:],
)
R.check(
    "setup: step 2 waits for step 1 and names the model size",
    not plugin.model_btn.isEnabled()
    and plugin.model_btn.text() == "Download model (%d MB)" % runtime.MODELS[runtime.detect()["model"]][2]
    and "Finish step 1" in plugin.model_btn.toolTip(),
    plugin.model_btn.text(),
)
plugin.model_box.setCurrentIndex(0)
plugin._model_picked(0)
R.check(
    "setup: picking another model in step 2 changes what will be downloaded, with a link to its page",
    plugin.model_btn.text() == "Download model (56 MB)"
    and QSettings().value("Segmentot/model") == "edge"
    and "href='https://huggingface.co/yonigozlan/EdgeTAM-hf'" in plugin.step2_text.text(),
    plugin.model_btn.text(),
)

# 2a. startup check: the message bar says what is still to do and opens the setup panel
plugin.dock.hide()
plugin._startup_check()
bar = iface.messageBar()
items = bar.items()
text = items[0].text() if items else ""
R.check(
    "startup: the message bar says the engine and the model are still to do",
    "install the engine" in text and "download the AI model" in text,
    text,
)
next(b for b in items[0].findChildren(QPushButton) if b.text() == "Open setup").click()
app.processEvents()
R.check("startup: 'Open setup' shows the panel and clears the message", not plugin.dock.isHidden() and not bar.items())

# 2b. permission guide: commands for this computer, copy buttons, and it opens when the folder is locked
g = setup_ui.GuideDialog(win)
cmds = [e.toPlainText() for e in g.findChildren(QPlainTextEdit)]
g.copy_buttons[0].click()
R.check(
    "guide: the first command is for this computer's engine folder and Copy puts it on the clipboard",
    runtime.home() in cmds[0]
    and QApplication.clipboard().text() == cmds[0]
    and g.copy_buttons[0].text().startswith("Copied"),
    cmds[0][:110],
)
locked = tempfile.mkdtemp()
os.chmod(locked, 0o555)
real_home = os.environ["SEGMENTOT_HOME"]
os.environ["SEGMENTOT_HOME"] = os.path.join(locked, "engine")
seen = press("Close")
plugin._install()
os.environ["SEGMENTOT_HOME"] = real_home
os.chmod(locked, 0o755)
shutil.rmtree(locked, True)
R.check(
    "guide: installing into a folder you cannot write to opens the guide instead of failing",
    "permission help" in seen.get("text", "")
    and plugin._setup_dlg is None
    and any(os.path.join(locked, "engine") in c for c in seen.get("commands", [])),
    seen.get("text", "not shown"),
)
fd = setup_ui.SetupDialog(win, "full", "base")
fd.show()
R.check("progress window: the log is folded away until asked for", fd.log.isHidden() and not fd.details.isChecked())
fd.task.error = "PermissionError: [Errno 13] Permission denied: '/x'"
fd._failed()
R.check(
    "guide: a permission error during setup offers 'Permission help…' and opens the details",
    not fd.guide_btn.isHidden() and "permission" in fd.head.text().lower() and not fd.log.isHidden(),
    fd.head.text()[:70],
)
fd.close()

# 2c. step 1 from a file (the model in the package is not the one picked above: the panel switches to it)
plugin_mod.QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (PACK, ""))
plugin._install_file()
dlg = plugin._setup_dlg
R.check(
    "setup: 'Install from a file…' opens the progress window and the step says it is working",
    dlg is not None
    and dlg.running
    and dlg.mode == "pack"
    and plugin.install_btn.text().startswith("Installing")
    and plugin.file_btn.isHidden()
    and not plugin.model_box.isEnabled(),
    dlg.windowTitle(),
)
t = pump(lambda: not dlg.running, 300, "the package install")
R.check(
    "setup: the package install finishes and its window closes by itself",
    runtime.is_ready() and "Done" in dlg.head.text() and dlg.isHidden(),
    "%.0f s · %s" % (t, dlg.head.text()[:60]),
)
R.check(
    "setup: with engine and model in place the panel moves to the work screen",
    plugin.setup_page.isHidden() and not plugin.work_page.isHidden() and QSettings().value("Segmentot/model") == "base",
    "model: %s" % QSettings().value("Segmentot/model"),
)
pump(lambda: "ready" in plugin.engine_label.text() or "problem" in plugin.engine_label.text(), 300, "the engine check")
R.check(
    "work screen: the engine starts by itself and the footer says so",
    "<b>ready</b>" in plugin.engine_label.text() and runtime.engine_running_here(),
    plugin.engine_label.text()[:110],
)
R.check(
    "work screen: without imagery, Start is off and the guide says to add a layer",
    not plugin.start_btn.isEnabled() and "imagery layer" in plugin.status.text(),
    plugin.status.text()[:60],
)

# 3. Settings: model list and "Save engine package…"
sd = setup_ui.EngineSettingsDialog(win, 8776, "base")
rows = {k: (t.text(), a.text(), d) for k, (t, a, d) in sd.rows.items()}
R.check(
    "settings: the Models tab lists every model with its size, what is downloaded and one action each",
    "✓ downloaded" in rows["base"][0]
    and rows["base"][1] == "✓ In use"
    and "not downloaded" in rows["tiny"][0]
    and rows["tiny"][1] == "Download (156 MB)"
    and "898 MB" in rows["large"][0]
    and list(rows)[0] == "edge"
    and rows["edge"][1] == "Download (56 MB)"
    and rows["sam3"][1] == "Download (3.4 GB)",
    " | ".join(r[1] for r in rows.values()),
)
R.check(
    "settings: every model row links to its documentation page",
    all("href='%s'" % runtime.model_docs(k) in r[0] for k, r in rows.items()),
    runtime.model_docs("sam3"),
)
R.check(
    "settings: only a downloaded model has a Delete button",
    [k for k, r in rows.items() if not r[2].isHidden()] == ["base"],
)
sd.rows["tiny"][1].click()
R.check(
    "settings: a model's Download button picks it and asks for the download",
    sd.values() == (8776, "tiny") and sd.download and sd.result() == 1,
    str(sd.values()),
)
sd = setup_ui.EngineSettingsDialog(win, 8776, "tiny")
sd.rows["base"][1].click()
R.check(
    "settings: 'Use' switches to a model that is already here without downloading",
    sd.rows["tiny"][1].text() == "Download (156 MB)" and sd.values() == (8776, "base") and not sd.download,
)

# 3b. a model that needs its maker's permission: the steps, the token, and the refusal message
td = setup_ui.TokenDialog(win, "sam3")
links = [b.toolTip() for b in td.findChildren(QPushButton) if b.toolTip().startswith("https://")]
R.check(
    "gated model: the dialog links to the model page and the token page, and waits for a token",
    links == ["https://huggingface.co/facebook/sam3", "https://huggingface.co/settings/tokens"]
    and not td.go.isEnabled()
    and td.go.text() == "Download (3.4 GB)",
    str(links),
)
td.token.setText("  hf_test_token  ")
td.accept()
with open(runtime.token_path()) as f:
    saved = f.read()
R.check(
    "gated model: the pasted token is saved for the download, readable by this user only",
    saved == "hf_test_token"
    and os.stat(runtime.token_path()).st_mode & 0o777 == 0o600
    and setup_ui.TokenDialog(win, "sam3").go.isEnabled(),
    runtime.token_path(),
)
os.remove(runtime.token_path())
fd = setup_ui.SetupDialog(win, "model", "sam3")
fd.show()
fd.task.error = (
    "OSError: You are trying to access a gated repo.\n401 Client Error.\nAccess to model facebook/sam3 is restricted."
)
fd._failed()
R.check(
    "gated model: a refused download says why and what to do",
    "refused the download" in fd.head.text() and "access" in fd.head.text() and fd.guide_btn.isHidden(),
    fd.head.text()[:60],
)
fd.close()

# 3c. delete a model to free its space (here the one in use: the panel goes back to step 2)
setup_ui.QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
plugin_mod.EngineSettingsDialog.exec = lambda self: (self._delete("base"), 0)[1]  # delete, then Cancel
plugin._engine_settings()
app.processEvents()
R.check(
    "delete: the model's files are gone and its engine is stopped",
    not runtime.model_downloaded("base") and not runtime.engine_running_here(),
)
R.check(
    "delete: after deleting the model in use the panel is back on step 2 with a Download button",
    not plugin.setup_page.isHidden()
    and plugin.work_page.isHidden()
    and plugin.model_btn.isEnabled()
    and plugin.model_btn.text() == "Download model (323 MB)"
    and plugin.install_btn.isHidden(),
    plugin.model_btn.text(),
)
sd = setup_ui.EngineSettingsDialog(win, 8776, "base")
R.check(
    "delete: the settings list shows it as not downloaded again, without a Delete button",
    "not downloaded" in sd.rows["base"][0].text()
    and sd.rows["base"][1].text() == "Download (323 MB)"
    and sd.rows["base"][2].isHidden(),
)
out = os.path.join(tempfile.mkdtemp(), "x")
setup_ui.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (out, ""))
setup_ui.QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.No)
sd._export()
R.check(
    "settings: 'Save engine package…' asks where and whether to include models",
    sd.export == (out + ".tar.gz", False),
    str(sd.export),
)
ex = setup_ui.SetupDialog(win, "export", "base", path=sd.export[0], with_model=sd.export[1])
ex.start()
t = pump(lambda: not ex.running, 600, "the export")
R.check(
    "settings: the package is written",
    os.path.exists(sd.export[0]) and "Done" in ex.head.text(),
    "%.0f MB in %.0f s" % (os.path.getsize(sd.export[0]) / 1e6 if os.path.exists(sd.export[0]) else 0, t),
)
plugin.unload()
shutil.rmtree(os.path.dirname(sd.export[0]), True)
if os.environ["SEGMENTOT_HOME"].startswith(tempfile.gettempdir()):
    shutil.rmtree(os.environ["SEGMENTOT_HOME"], True)  # finish() exits without running atexit
e2e.finish(R, app)
