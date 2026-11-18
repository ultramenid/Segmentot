"""Qt6 check that needs no Qt6 build of QGIS: only a Python with PyQt6 (pip install PyQt6).

1. Static: every Qt name the plugin code uses (Qt.AlignmentFlag.AlignLeft, QMessageBox.StandardButton.Yes ..)
   must exist in PyQt6, where an enum value is only reachable through its enum.
2. Run: the whole panel and every dialog are built and driven on real Qt 6 widgets, with stand-ins for the
   QGIS classes (map canvas, layer combo box, tasks ..).

It does NOT check the QGIS API of QGIS 4 itself: only the tests run with a Qt6 QGIS do that.
Run:  python tests/qt6_check.py
"""
import ast
import enum
import importlib
import os
import sys
import tempfile
import types
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("SEGMENTOT_HOME", tempfile.mkdtemp(prefix="segmentot_qt6_"))      # a computer without engine
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("SEGMENTOT_TEST_PARENT") or os.path.dirname(REPO))

from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print("{}  {}{}".format("PASS" if ok else "FAIL", name, "  — " + str(detail) if detail else ""), flush=True)


# ------------------------------------------------------------------ 1. static
MOVED = {"QAction": "QtGui", "QActionGroup": "QtGui", "QShortcut": "QtGui"}     # qgis.PyQt.QtWidgets still has them


def unknown_qt_names(path):
    """Names like Qt.AlignLeft that PyQt6 does not have (line, name)."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    known, bad = {}, []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("qgis.PyQt."):
            for a in node.names:
                mod = importlib.import_module("PyQt6." + MOVED.get(a.name, node.module.split(".")[-1]))
                if hasattr(mod, a.name):
                    known[a.asname or a.name] = getattr(mod, a.name)
                else:
                    bad.append((node.lineno, "import " + a.name))

    class Visitor(ast.NodeVisitor):
        def visit_Attribute(self, node):
            chain, root = [], node
            while isinstance(root, ast.Attribute):
                chain.append(root.attr)
                root = root.value
            if not (isinstance(root, ast.Name) and root.id in known):
                return self.generic_visit(node)
            obj = known[root.id]
            for attr in reversed(chain):
                if not hasattr(obj, attr):
                    bad.append((node.lineno, root.id + "." + ".".join(reversed(chain))))
                    break
                obj = getattr(obj, attr)
                if not isinstance(obj, type):       # a method, signal or enum value: nothing deeper to resolve
                    break
    Visitor().visit(tree)
    return bad


for name in ("gui/plugin.py", "gui/setup_ui.py", "gui/click_tool.py", "core/engine.py"):
    bad = unknown_qt_names(os.path.join(REPO, name))
    check("static: {} uses only Qt names that PyQt6 has".format(name), not bad, bad[:6])


# ------------------------------------------------------------------ 2. stand-ins for QGIS, then the real UI
class AnyModule(types.ModuleType):
    """A QGIS module where every class not defined below accepts anything."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = mock.MagicMock(name=name)
        setattr(self, name, value)
        return value


class QgsTask(QtCore.QObject):
    progressChanged = QtCore.pyqtSignal(float)
    taskCompleted = QtCore.pyqtSignal()
    taskTerminated = QtCore.pyqtSignal()

    class Flag(enum.Flag):
        CanCancel = 2
        Hidden = 8

    def __init__(self, description="", flags=None):
        super().__init__()

    def isCanceled(self):
        return False

    def cancel(self):
        pass

    fromFunction = staticmethod(lambda *a, **k: mock.MagicMock())


class QgsMapTool(QtCore.QObject):
    deactivated = QtCore.pyqtSignal()

    def __init__(self, canvas):
        super().__init__()

    def setCursor(self, cursor):
        pass


class QgsMapLayerComboBox(QtWidgets.QComboBox):
    layerChanged = QtCore.pyqtSignal(object)
    _layer = None

    def setFilters(self, filters):
        pass

    def setAllowEmptyLayer(self, allow, text=""):
        self.addItem(text)

    def currentLayer(self):
        return self._layer

    def setLayer(self, layer):
        self._layer = layer
        self.layerChanged.emit(layer)


class QgsCollapsibleGroupBox(QtWidgets.QGroupBox):
    def setCollapsed(self, collapsed):
        pass


core, gui = AnyModule("qgis.core"), AnyModule("qgis.gui")
core.QgsTask = QgsTask
gui.QgsMapTool, gui.QgsMapLayerComboBox = QgsMapTool, QgsMapLayerComboBox
gui.QgsCollapsibleGroupBox = QgsCollapsibleGroupBox
widgets = types.ModuleType("qgis.PyQt.QtWidgets")
widgets.__dict__.update(vars(QtWidgets))
widgets.QAction = QtGui.QAction                     # as QGIS's own Qt6 layer does
sys.modules.update({"qgis": AnyModule("qgis"), "qgis.core": core, "qgis.gui": gui, "qgis.PyQt": AnyModule("qgis.PyQt"),
                    "qgis.PyQt.QtCore": QtCore, "qgis.PyQt.QtGui": QtGui, "qgis.PyQt.QtWidgets": widgets})

app = QtWidgets.QApplication([])
app.setOrganizationName("SegmentotE2E")             # keep the test's QSettings away from the real QGIS profile
app.setApplicationName("SegmentotQt6")
QtCore.QSettings().setValue("Segmentot/port", 8794)
from Segmentot.core import runtime  # noqa: E402
from Segmentot.gui import plugin as plugin_mod, setup_ui  # noqa: E402

win = QtWidgets.QMainWindow()
win.resize(1200, 800)
win.show()
canvas = mock.MagicMock(name="canvas")
iface = mock.MagicMock(name="iface")
iface.mainWindow.return_value, iface.mapCanvas.return_value = win, canvas
iface.addDockWidget.side_effect = win.addDockWidget
try:
    plugin = plugin_mod.SegmentotPlugin(iface)
    plugin.initGui()
    plugin.dock.show()
    app.processEvents()
    check("run: the panel builds and shows the setup steps on Qt " + QtCore.QT_VERSION_STR,
          not plugin.setup_page.isHidden() and plugin.work_page.isHidden()
          and plugin.step1.title() == "Step 1 · Install the engine" and plugin.model_btn.text().startswith("Download"),
          plugin.model_btn.text())
    plugin.model_box.setCurrentIndex(0)
    plugin._model_picked(0)
    check("run: picking a model in step 2 works", plugin.model_btn.text() == "Download model (56 MB)")
    plugin._startup_check()
    check("run: the startup message is put in the message bar", iface.messageBar.return_value.pushWidget.called)

    plugin._ready = lambda: True                    # as if engine and model were installed
    plugin._check_engine = lambda: None
    plugin._refresh()
    check("run: with engine and model the work screen shows, Start waits for imagery",
          plugin.setup_page.isHidden() and not plugin.work_page.isHidden() and not plugin.start_btn.isEnabled())
    plugin.raster.setLayer(mock.MagicMock(name="raster layer"))
    plugin.start_btn.setChecked(True)
    check("run: Start hands the map tool to the canvas and switches the action buttons on",
          canvas.setMapTool.called and all(b.isEnabled() for b in plugin.actions) and not plugin.keys.isHidden())
    plugin.square.setChecked(True)
    plugin.outline.setCurrentIndex(1)
    plugin.start_btn.setChecked(False)
    check("run: Stop switches them off again", not any(b.isEnabled() for b in plugin.actions))

    guide = setup_ui.GuideDialog(win)
    guide.copy_buttons[0].click()
    check("run: permission guide opens and Copy works", guide.copy_buttons[0].text().startswith("Copied"))
    token = setup_ui.TokenDialog(win, "sam3")
    token.token.setText("hf_x")
    check("run: access dialog for a gated model", token.go.isEnabled() and token.go.text() == "Download (3.4 GB)")
    settings = setup_ui.EngineSettingsDialog(win, 8776, "base")
    settings.show()
    settings.rows["tiny"][1].click()
    check("run: settings dialog (Models and Engine tabs), picking a model",
          len(settings.rows) == len(runtime.MODELS) and settings.values() == (8776, "tiny"))
    progress = setup_ui.SetupDialog(win, "full", "base")
    progress.show()
    progress._line("Downloading torch")
    progress.task.error = "PermissionError: [Errno 13] Permission denied"
    progress._failed()
    check("run: progress window, its details and a failure message",
          not progress.log.isHidden() and not progress.guide_btn.isHidden() and "permission" in progress.head.text())
    for d in (guide, token, settings, progress):
        d.close()
    plugin.unload()
    check("run: the plugin unloads", iface.removeDockWidget.called)
except Exception:
    import traceback
    check("run: no exception on Qt 6", False, traceback.format_exc()[-1500:])

print("\n{} checks, {} failed".format(len(RESULTS), RESULTS.count(False)), flush=True)
sys.stdout.flush()
os._exit(1 if False in RESULTS else 0)              # skip Qt teardown, like the other headless tests
