import datetime
import html
import os
import shutil

from qgis.core import (
    QgsApplication,
    QgsCoordinateTransform,
    QgsCoordinateTransformContext,
    QgsCsException,
    QgsFeature,
    QgsGeometry,
    QgsMapLayerProxyModel,
    QgsProject,
    QgsTask,
    QgsVectorFileWriter,
    QgsVectorLayer,
    QgsWkbTypes,
)
from qgis.gui import QgsCollapsibleGroupBox, QgsMapLayerComboBox
from qgis.PyQt.QtCore import QSettings, Qt, QTimer
from qgis.PyQt.QtGui import QColor, QIcon
from qgis.PyQt.QtWidgets import (
    QAction, QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from ..core import engine, runtime
from .click_tool import SegmentotTool
from .setup_ui import PACK_FILTER, EngineSettingsDialog, GuideDialog, SetupDialog, TokenDialog

# the fields of a new layer, as memory-provider text: the same on Qt5 and Qt6 (no QVariant types needed)
FIELDS = "&field=class:string&field=score:double&field=area_m2:double&field=created:string"
PRIMARY = "QPushButton{font-weight:bold;padding:6px}"       # the one button to press next


def slim(combo):
    """Let a combo box shrink with a narrow panel instead of forcing a horizontal scrollbar."""
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(6)
    return combo


class SegmentotPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.canvas = iface.mapCanvas()
        self.dock = None
        self._setup_dlg = None
        self._info = None         # runtime.detect(): looked up the first time the setup steps are shown
        self._engine_ok = None    # None: not started yet · True: answers · False: failed (no automatic retry)
        self._starting = False
        self.saved = []           # (layer id, fid, added as an edit?) for undo

    # ------------------------------------------------------------------ setup
    def initGui(self):
        self.action = QAction(QIcon(os.path.join(runtime.PLUGIN_DIR, "resources", "icon.svg")),
                              "Segmentot: click an object to outline it", self.iface.mainWindow())
        self.action.setCheckable(True)
        self.action.triggered.connect(self._toggle)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToRasterMenu("Segmentot", self.action)

        self.tool = SegmentotTool(self.canvas, self._settings, self._save)
        self.tool.undoSaved.connect(self._undo_saved)
        self.tool.status.connect(lambda s: self.status.setText(s))
        self.tool.deactivated.connect(lambda: self.start_btn.setChecked(False))
        self._build_dock()
        self._startup = QTimer()                  # after QGIS has finished starting
        self._startup.setSingleShot(True)
        self._startup.timeout.connect(self._startup_check)
        self._startup.start(2000)

    def unload(self):
        self._startup.stop()
        self.canvas.destinationCrsChanged.disconnect(self._new_imagery)
        runtime.stop_engine()
        if self.canvas.mapTool() is self.tool:
            self.canvas.unsetMapTool(self.tool)
        self.iface.removeToolBarIcon(self.action)
        self.iface.removePluginRasterMenu("Segmentot", self.action)
        self.iface.removeDockWidget(self.dock)
        self.dock.deleteLater()

    def _toggle(self, on):
        self.dock.setVisible(on)
        if on:
            self.dock.raise_()
        else:
            self.start_btn.setChecked(False)      # a hidden panel must not keep saving clicks

    def _shown(self, visible):
        self.action.setChecked(visible)
        if visible:
            self._refresh()
            if self._engine_ok is None:
                self._check_engine()

    # ------------------------------------------------------------------ the panel
    def _build_dock(self):
        """One panel, two screens: the setup steps until everything is installed, then the work screen."""
        self.dock = QDockWidget("Segmentot", self.iface.mainWindow())
        self.dock.setObjectName("SegmentotDock")
        root = QWidget()
        lay = QVBoxLayout(root)
        self.setup_page = self._setup_page()
        self.work_page = self._work_page()
        lay.addWidget(self.setup_page)
        lay.addWidget(self.work_page)
        lay.addStretch(1)
        foot = QHBoxLayout()
        self.engine_label = QLabel("")
        self.engine_label.setWordWrap(True)
        self.engine_label.linkActivated.connect(lambda _link: self._check_engine())      # "Try again"
        foot.addWidget(self.engine_label, 1)
        bs = QPushButton("Settings…")
        bs.setToolTip("Model, engine folder, reinstall or remove the engine, save an engine package")
        bs.clicked.connect(self._engine_settings)
        foot.addWidget(bs, 0, Qt.AlignmentFlag.AlignBottom)
        lay.addLayout(foot)

        scroll = QScrollArea()             # small laptop screens: the panel scrolls instead of clipping
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(root)
        self.dock.setWidget(scroll)
        self.iface.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
        self.dock.hide()
        self.dock.visibilityChanged.connect(self._shown)
        self._refresh()

    def _setup_page(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        head = QLabel("<b>Set up Segmentot</b><br>Two one-time steps, then you outline objects by clicking them. "
                      "Your imagery never leaves this computer.")
        head.setWordWrap(True)
        v.addWidget(head)

        self.step1 = QGroupBox()
        s = QVBoxLayout(self.step1)
        self.step1_text = QLabel()
        self.step1_text.setWordWrap(True)
        s.addWidget(self.step1_text)
        self.install_btn = QPushButton()
        self.install_btn.setStyleSheet(PRIMARY)
        self.install_btn.clicked.connect(lambda: self._install())
        s.addWidget(self.install_btn)
        self.file_btn = QPushButton("No internet? Install from a file…")
        self.file_btn.setToolTip("Use an engine package file made on another computer\n"
                                 "(there: Settings… → Save engine package…)")
        self.file_btn.clicked.connect(lambda: self._install_file())
        s.addWidget(self.file_btn)
        v.addWidget(self.step1)

        self.step2 = QGroupBox()
        s = QVBoxLayout(self.step2)
        self.step2_text = QLabel()
        self.step2_text.setWordWrap(True)
        self.step2_text.setOpenExternalLinks(True)       # "About this model"
        s.addWidget(self.step2_text)
        self.model_box = slim(QComboBox())
        self.model_box.setToolTip("Bigger models follow edges more exactly and are slower")
        self.model_box.activated.connect(self._model_picked)      # activated: only when the user picks
        s.addWidget(self.model_box)
        self.model_btn = QPushButton()
        self.model_btn.setStyleSheet(PRIMARY)
        self.model_btn.clicked.connect(lambda: self._download_model())
        s.addWidget(self.model_btn)
        v.addWidget(self.step2)

        bh = QPushButton("Permission help…")
        bh.setToolTip("Ready-to-paste commands for when the install needs administrator rights")
        bh.clicked.connect(lambda: GuideDialog(self.dock).exec())
        v.addWidget(bh, 0, Qt.AlignmentFlag.AlignLeft)
        return page

    def _work_page(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)

        v.addWidget(QLabel("<b>1 · Imagery</b>"))
        self.raster = slim(QgsMapLayerComboBox())
        self.raster.setFilters(QgsMapLayerProxyModel.Filter.RasterLayer)
        self.raster.setToolTip("The picture you click on: a GeoTIFF, an orthophoto, or an online layer")
        v.addWidget(self.raster)

        v.addWidget(QLabel("<b>2 · Save to</b>"))
        row = QHBoxLayout()
        self.target = slim(QgsMapLayerComboBox())
        self.target.setFilters(QgsMapLayerProxyModel.Filter.PolygonLayer | QgsMapLayerProxyModel.Filter.WritableLayer)
        self.target.setAllowEmptyLayer(True, "New temporary layer")
        self.target.setToolTip("Polygon layer the objects go to.\n"
                               "A temporary layer is lost when QGIS closes unless you export it.\n"
                               "If the layer is in edit mode, objects are added as edits (QGIS undo works);\n"
                               "otherwise they are written straight to the layer.")
        row.addWidget(self.target, 1)
        bg = QPushButton("New file…")
        bg.setToolTip("Create a GeoPackage file and save the objects there, so they are kept on disk")
        bg.clicked.connect(self._new_gpkg_layer)
        row.addWidget(bg)
        v.addLayout(row)
        self.cls = QLineEdit(QSettings().value("Segmentot/class", ""))
        self.cls.textChanged.connect(lambda t: QSettings().setValue("Segmentot/class", t))
        self.cls.returnPressed.connect(lambda: self.canvas.setFocus())
        v.addWidget(self.cls)
        self.target.layerChanged.connect(self._target_changed)
        self._target_changed(self.target.currentLayer())

        v.addWidget(QLabel("<b>3 · Click the objects</b>"))
        self.start_btn = QPushButton("▶  Start")
        self.start_btn.setCheckable(True)
        self.start_btn.setStyleSheet(PRIMARY)
        self.start_btn.toggled.connect(self._start_stop)
        v.addWidget(self.start_btn)
        self.status = QLabel("")           # the guide: always says what to do next
        self.status.setWordWrap(True)
        self.status.setFrameShape(QFrame.Shape.StyledPanel)
        self.status.setMargin(6)
        self.status.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        # room for the longest everyday message: the panel below must not jump at every click
        self.status.setMinimumHeight(self.status.fontMetrics().lineSpacing() * 4 + 16)
        v.addWidget(self.status)
        grid = QGridLayout()
        self.actions = []
        for text, fn, tip in (("Save", lambda: self.tool.accept(), "Enter"),
                              ("Undo", lambda: self.tool.undo(), "Backspace: last point, or last saved object"),
                              ("Other outline", lambda: self.tool.next_candidate(), "C: try another outline"),
                              ("Clear", lambda: self.tool.clear(), "Esc")):
            bb = QPushButton(text)
            bb.setToolTip(tip)
            bb.setFocusPolicy(Qt.FocusPolicy.NoFocus)     # keep keyboard focus on the map for the shortcuts
            bb.setEnabled(False)
            bb.clicked.connect(fn)
            grid.addWidget(bb, len(self.actions) // 2, len(self.actions) % 2)
            self.actions.append(bb)
        v.addLayout(grid)
        self.keys = QLabel("<small>Left-click adds · right-click removes · <b>Enter</b> saves · <b>Backspace</b> "
                           "undoes · <b>C</b> other outline · <b>Esc</b> clears.<br>A left-click outside the "
                           "dotted frame saves the object and starts a new one.</small>")
        self.keys.setWordWrap(True)
        self.keys.setVisible(False)
        v.addWidget(self.keys)

        # --- everything optional, folded away
        obox = QgsCollapsibleGroupBox("More options")
        obox.setObjectName("SegmentotOptions")
        obox.setCollapsed(True)
        f = QFormLayout(obox)
        self.detail = slim(QComboBox())
        self.detail.addItems(["Raster pixels (sharpest)", "Map zoom (as on screen)"])
        self.detail.setToolTip("Raster pixels: outlines sit exactly on the raster's pixel edges (local files).\n"
                               "Map zoom: what you see is what the AI sees. Zoom out for big objects\n"
                               "(fields, lakes), zoom in for small ones. Always used for online layers.")
        f.addRow("Detail", self.detail)
        self.outline = slim(QComboBox())
        self.outline.addItems(["Follow pixel edges", "Smooth"])
        f.addRow("Outline", self.outline)
        self.hover = QCheckBox("Preview under the cursor")
        self.hover.setChecked(QSettings().value("Segmentot/hover", True, type=bool))
        self.hover.toggled.connect(lambda on: QSettings().setValue("Segmentot/hover", on))
        f.addRow(self.hover)
        self.fill_holes = QCheckBox("Fill holes")
        self.square = QCheckBox("Square corners (buildings)")
        self.smooth = QCheckBox("Smooth curves (fields, water)")
        for w in (self.fill_holes, self.square, self.smooth):
            f.addRow(w)
        self.expand = QSpinBox()
        self.expand.setRange(-20, 20)
        self.expand.setSuffix(" px")
        self.expand.setToolTip("Grow (+) or shrink (−) the outline, in image pixels")
        f.addRow("Grow / shrink", self.expand)
        self.simplify = QDoubleSpinBox()
        self.simplify.setRange(0, 20)
        self.simplify.setSingleStep(0.5)
        self.simplify.setSuffix(" px")
        self.simplify.setToolTip("Fewer corner points: higher = simpler outline")
        f.addRow("Simplify", self.simplify)
        self.min_area = QDoubleSpinBox()
        self.min_area.setRange(0, 1e7)
        self.min_area.setDecimals(1)
        self.min_area.setSuffix(" m²")
        self.min_area.setToolTip("Drop separate pieces smaller than this")
        f.addRow("Drop pieces under", self.min_area)
        for w in (self.expand, self.simplify, self.min_area):
            w.valueChanged.connect(lambda *a: self.tool.refresh())
        for w in (self.fill_holes, self.square, self.smooth):
            w.toggled.connect(lambda *a: self.tool.refresh())
        v.addWidget(obox)

        for sig in (self.raster.layerChanged, self.detail.currentIndexChanged, self.canvas.destinationCrsChanged):
            sig.connect(self._new_imagery)
        self.outline.currentIndexChanged.connect(lambda *a: self.tool.resend())
        self._idle_hint()
        return page

    def _new_imagery(self, *_):
        self.tool.new_imagery()
        self._idle_hint()

    def _idle_hint(self):
        """While not clicking, the guide box says the next thing to do."""
        have = self.raster.currentLayer() is not None
        if not have:
            self.start_btn.setChecked(False)
        self.start_btn.setEnabled(have)
        if not self.start_btn.isChecked():
            self.status.setText("Press Start, then click an object on the map." if have else
                                "Add an imagery layer to the map first: a GeoTIFF or orthophoto, "
                                "or an online layer such as a satellite basemap.")

    # ------------------------------------------------------------------ settings for the tool
    def _settings(self):
        layer = self.raster.currentLayer()
        crs = self.canvas.mapSettings().destinationCrs()
        grid = engine.native_grid(layer, crs) if self.detail.currentIndex() == 0 else None
        return {
            "layer": layer,
            "grid": grid,
            "outline": "pixel" if self.outline.currentIndex() == 0 else "smooth",
            "hover": self.hover.isChecked(),
            "refine": {"expand": self.expand.value(), "simplify": self.simplify.value(),
                       "min_area_m2": self.min_area.value(), "fill_holes": self.fill_holes.isChecked(),
                       "square": self.square.isChecked(), "smooth": self.smooth.isChecked()},
        }

    def _start_stop(self, on):
        if on and not (self._ready() and self.raster.currentLayer() is not None):
            self.start_btn.setChecked(False)
            self._refresh()
            return
        for b in self.actions:
            b.setEnabled(on)
        self.keys.setVisible(on)
        self.start_btn.setText("■  Stop" if on else "▶  Start")
        if on:
            if self.detail.currentIndex() == 0 and self._settings()["grid"] is None:
                self.status.setText("Hover over an object, then left-click it.\n(Using the map zoom: raster pixels "
                                    "need a local raster in the project's CRS, with square pixels.)")
            else:
                self.status.setText("Hover over an object, then left-click it.")
            self.canvas.setMapTool(self.tool)
        else:
            try:
                if self.canvas.mapTool() is self.tool:
                    self.canvas.unsetMapTool(self.tool)
            except RuntimeError:      # canvas already gone (QGIS closing)
                pass
            self._idle_hint()

    # ------------------------------------------------------------------ engine and setup steps
    def _engine_cfg(self):
        st = runtime.load_state()
        port = int(QSettings().value("Segmentot/port", 8776))
        model = QSettings().value("Segmentot/model", "") or st.get("model") or (self._info or {}).get("model")
        return port, model if model in runtime.MODELS else "base"

    def _force_cpu(self):
        return runtime.EDITION == "cpu" or QSettings().value("Segmentot/force_cpu", False, type=bool)

    def _engine_answers(self):
        try:
            return bool(engine._call("/health", timeout=1).get("ok"))
        except Exception:
            return False

    def _ready(self):
        """Engine and model are on this computer (or an engine is already running)."""
        return bool(runtime.is_ready() and runtime.model_downloaded(self._engine_cfg()[1])) or self._engine_answers()

    def _refresh(self, *_):
        """Show the setup steps until the engine and its model are here, then the work screen."""
        engine.configure(*self._engine_cfg())
        ready = self._ready()
        self.setup_page.setVisible(not ready)
        self.work_page.setVisible(ready)
        if not ready:
            self._engine_ok = None
            self.start_btn.setChecked(False)
            self.engine_label.setText("")
            if self.dock.isVisible():           # not at QGIS start: looking for the GPU takes a moment
                self._fill_setup()

    def _fill_setup(self):
        if self._info is None:
            self._info = runtime.detect()
            engine.configure(*self._engine_cfg())       # the model that fits this computer
        info, model = self._info, self._engine_cfg()[1]
        busy = self._setup_dlg.mode if self._setup_dlg is not None and self._setup_dlg.running else None
        installed, have_model = runtime.is_ready(), runtime.model_downloaded(model)

        if installed:
            t = runtime.load_state().get("torch", {})
            self.step1.setTitle("✓ Step 1 · Engine installed")
            self.step1_text.setText("Runs on: <b>{}</b>".format(
                t.get("gpu") or ("Apple GPU" if t.get("mps") else "CPU")))
        else:
            gpu = info["accel"] != "cpu" and not self._force_cpu()
            try:
                free = "{:.0f} GB".format(shutil.disk_usage(os.path.dirname(runtime.home()) or ".").free / 1e9)
            except OSError:
                free = "unknown"
            text = ("Segmentot's own Python and PyTorch. It is kept apart from QGIS and can be removed in Settings."
                    "<br>Will run on: <b>{}</b><br>Download: about {} · free disk space: {}").format(
                {"cuda": "NVIDIA GPU", "mps": "Apple GPU (Metal)"}.get(info["accel"], "CPU") if gpu else "CPU",
                "4–5 GB" if info["nvidia"] and not self._force_cpu() else "1–2 GB", free)
            if info["notes"]:
                text += "<br><small>" + "<br>".join(n.replace("\n", "<br>") for n in info["notes"]) + "</small>"
            self.step1.setTitle("Step 1 · Install the engine")
            self.step1_text.setText(text)
        self.install_btn.setVisible(not installed)
        self.install_btn.setText("Installing… show progress" if busy in ("full", "pack") else "Install engine")
        self.file_btn.setVisible(not installed and busy is None)

        self.model_box.clear()
        for k, (_id, label, mb) in runtime.MODELS.items():
            self.model_box.addItem("{} · {}".format(label, "✓ downloaded" if runtime.model_downloaded(k)
                                                    else runtime.size_text(mb)), k)
        self.model_box.setCurrentIndex(self.model_box.findData(model))
        self.model_box.setEnabled(busy is None)
        self.step2.setTitle("✓ Step 2 · AI model downloaded" if have_model else "Step 2 · Download the AI model")
        self.step2_text.setText("The AI that finds the outlines. It is downloaded from huggingface.co{} "
                                "<a href='{}'>About this model</a>".format(
                                    "." if installed else ", after step 1.", runtime.model_docs(model)))
        self.model_btn.setVisible(not have_model)
        self.model_btn.setEnabled(installed and busy in (None, "model"))
        self.model_btn.setToolTip("" if installed else "Finish step 1 first")
        self.model_btn.setText("Downloading… show progress" if busy == "model" else
                               "Download model ({})".format(runtime.size_text(runtime.MODELS[model][2])))

    def _model_picked(self, _index):
        QSettings().setValue("Segmentot/model", self.model_box.currentData())
        self._refresh()

    def _startup_check(self):
        """Once per QGIS start: say in the message bar what is still to do before the first click."""
        todo = ([] if runtime.is_ready() else ["install the engine"]) + \
            ([] if runtime.model_downloaded(self._engine_cfg()[1]) else ["download the AI model"])
        if not todo:
            return
        bar = self.iface.messageBar()
        item = bar.createMessage("Segmentot", "One-time setup needed before the first use: {}.".format(
            " and ".join(todo)))
        b = QPushButton("Open setup")

        def go():
            bar.popWidget(item)
            self.dock.show()
            self.dock.raise_()
        b.clicked.connect(go)
        item.layout().addWidget(b)
        bar.pushWidget(item)

    def _setup_running(self):
        """Show the open install/download window instead of starting a second setup in the same folder."""
        d = self._setup_dlg
        if d is None or not d.running:
            return False
        d.show()
        d.raise_()
        d.activateWindow()
        return True

    def _run_setup(self, mode, model, path=None):
        self._setup_dlg = SetupDialog(self.iface.mainWindow(), mode, model, self._force_cpu(), path)
        self._setup_dlg.finishedOk.connect(self._installed)
        self._setup_dlg.failed.connect(self._refresh)
        self._setup_dlg.start()
        self._refresh()

    def _install(self, pack=None):
        """Step 1: download and install the engine, or install it from the engine package file `pack`."""
        if self._setup_running():
            return
        if not runtime.home_writable():
            GuideDialog(self.dock, "<b>Segmentot cannot write to its engine folder.</b> Run the first command "
                                   "below once, then press <i>Install engine</i> again.").exec()
            return
        model = self._engine_cfg()[1]
        QSettings().setValue("Segmentot/model", model)
        runtime.stop_engine()
        self._run_setup("pack" if pack else "full", model, pack)

    def _install_file(self):
        path, _ = QFileDialog.getOpenFileName(self.dock, "Segmentot engine package", os.path.expanduser("~"),
                                              PACK_FILTER)
        if path:
            self._install(path)

    def _download_model(self):
        """Step 2. Only ever started by this button: the engine never downloads a model on its own."""
        if self._setup_running():
            return
        model = self._engine_cfg()[1]
        QSettings().setValue("Segmentot/model", model)
        self._get_model(model)

    def _get_model(self, model):
        """Download a model; one that needs its maker's permission first shows the steps and takes the token."""
        if model in runtime.GATED and not TokenDialog(self.dock, model).exec():
            return self._refresh()
        self._run_setup("model", model)

    def _installed(self):
        if not runtime.model_downloaded(self._engine_cfg()[1]):
            have = [k for k in runtime.MODELS if runtime.model_downloaded(k)]
            if have:                            # the engine package came with another model: use that one
                packed = runtime.load_state().get("model")
                QSettings().setValue("Segmentot/model", packed if packed in have else have[-1])
        self._engine_ok = None
        self.dock.show()
        self.dock.raise_()
        self._shown(True)

    def _engine_settings(self):
        if self._setup_running():       # no port/model change or "Remove engine" while a setup runs
            return
        port, model = self._engine_cfg()
        dlg = EngineSettingsDialog(self.dock, port, model)
        if not dlg.exec():
            if dlg.deleted:             # e.g. the model in use: back to step 2
                self._engine_ok = None
                self._shown(self.dock.isVisible())
            return
        if dlg.export:
            self._setup_dlg = SetupDialog(self.iface.mainWindow(), "export", model, path=dlg.export[0],
                                          with_model=dlg.export[1])
            self._setup_dlg.start()
            return
        new_port, new_model = dlg.values()
        QSettings().setValue("Segmentot/port", new_port)
        QSettings().setValue("Segmentot/model", new_model)
        QSettings().setValue("Segmentot/force_cpu", dlg.force_cpu.isChecked())
        if dlg.reinstall:
            return self._install()
        if not dlg.removed and (new_port, new_model) != (port, model):
            runtime.stop_engine()
            engine.configure(new_port, new_model)
            self.tool.new_imagery()
        if dlg.download:
            return self._get_model(new_model)
        self._engine_ok = None
        self._shown(self.dock.isVisible())

    def _check_engine(self):
        """Start the engine in the background; the footer says how it went."""
        if self._starting or not self._ready():
            return
        self._starting = True
        self.engine_label.setText("<small>Engine: starting (loading the model takes 10–30 s the first time)…</small>")

        def done(exc, r=None):
            self._starting = False
            if exc is not None:
                self._engine_ok = False
                self.engine_label.setText("<b>Engine problem:</b> {} <a href='retry'>Try again</a>".format(
                    html.escape(str(exc)).replace("\n", "<br>")))
                return
            self._engine_ok = True
            st = runtime.load_state()
            where = "" if runtime.engine_running_here() else " (already running engine)"
            dev = "CPU" if r.get("device") == "cpu" else r.get("gpu") or "CPU"   # "cpu" after a GPU fallback
            lines = ["Engine <b>ready</b>{} · {} · {}".format(where, dev, r.get("model", ""))]
            if r.get("notice"):
                lines.append(html.escape(r["notice"]))
            lines += [n.replace("\n", "<br>") for n in st.get("detected", {}).get("notes", [])
                      if "flatpak override" in n and not st.get("torch", {}).get("cuda")]
            self.engine_label.setText("<small>" + "<br>".join(lines) + "</small>")
        self._health = QgsTask.fromFunction(
            "Segmentot engine", lambda t: engine.call("/health", timeout=600, headers={"X-Segmentot": "load"}),
            on_finished=done)
        QgsApplication.taskManager().addTask(self._health)

    # ------------------------------------------------------------------ output
    def _style(self, lyr):
        sym = lyr.renderer().symbol()
        sym.setColor(QColor(255, 196, 0, 60))
        sym.symbolLayer(0).setStrokeColor(QColor(255, 150, 0))
        sym.symbolLayer(0).setStrokeWidth(0.5)

    def _new_memory_layer(self, add=True):
        crs = self.canvas.mapSettings().destinationCrs()
        lyr = QgsVectorLayer("MultiPolygon?crs={}{}".format(crs.authid() or crs.toWkt(), FIELDS), "Segmentot",
                             "memory")
        self._style(lyr)
        if add:
            QgsProject.instance().addMapLayer(lyr)
        return lyr

    def _new_gpkg_layer(self):
        path, _ = QFileDialog.getSaveFileName(self.dock, "New Segmentot GeoPackage",
                                              os.path.expanduser("~/segmentot.gpkg"), "GeoPackage (*.gpkg)")
        if not path:
            return
        if not path.lower().endswith(".gpkg"):
            path += ".gpkg"
        tmp = self._new_memory_layer(add=False)     # only a template: the project deletes layers it removes
        name = os.path.splitext(os.path.basename(path))[0]
        opts = QgsVectorFileWriter.SaveVectorOptions()
        opts.driverName = "GPKG"
        opts.layerName = name
        if os.path.exists(path):
            opts.actionOnExistingFile = QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteLayer
        err = QgsVectorFileWriter.writeAsVectorFormatV3(tmp, path, QgsCoordinateTransformContext(), opts)
        if err[0] != QgsVectorFileWriter.WriterError.NoError:
            QMessageBox.warning(self.dock, "Segmentot", "Could not create it: {}".format(err[1]))
            return
        lyr = QgsVectorLayer("{}|layername={}".format(path, name), name, "ogr")
        self._style(lyr)
        QgsProject.instance().addMapLayer(lyr)
        self.target.setLayer(lyr)

    def _target_changed(self, lyr):
        ok = lyr is None or lyr.fields().indexFromName("class") >= 0    # None = new temporary layer, has it
        self.cls.setEnabled(ok)
        self.cls.setPlaceholderText("Class of the objects, e.g. building (optional)" if ok
                                    else "This layer has no 'class' field")
        self.cls.setToolTip("Written to the 'class' field of every object you save")

    def _warn(self, msg):
        self.status.setText(msg)
        self.iface.messageBar().pushWarning("Segmentot", msg)
        return False

    def _save(self, geom, score):
        """Store the object. False (with a warning) when it could not be written."""
        lyr = self.target.currentLayer()
        if lyr is None:
            lyr = self._new_memory_layer()
            self.target.setLayer(lyr)
        canvas_crs = self.canvas.mapSettings().destinationCrs()
        g = QgsGeometry(engine.polygons_only(geom))       # a copy: the tool keeps geom if saving fails
        if QgsWkbTypes.isMultiType(lyr.wkbType()):
            g.convertToMultiType()
        elif g.isMultipart():
            parts = g.asGeometryCollection()
            g = max(parts, key=lambda p: p.area())        # single-part layer: keep the main piece
        area = engine.area_m2(g, canvas_crs)
        if lyr.crs() != canvas_crs:
            try:
                g.transform(QgsCoordinateTransform(canvas_crs, lyr.crs(), QgsProject.instance()))
            except QgsCsException as e:
                return self._warn("Could not convert the object to the CRS of '{}': {}".format(lyr.name(), e))
        f = QgsFeature(lyr.fields())
        f.setGeometry(g)
        vals = {"class": self.cls.text().strip() or None, "score": round(float(score), 4),
                "area_m2": round(area, 2), "created": datetime.datetime.now().isoformat(timespec="seconds")}
        for name, v in vals.items():
            i = lyr.fields().indexFromName(name)
            if i >= 0:
                f.setAttribute(i, v)
        if lyr.isEditable():
            lyr.beginEditCommand("Segmentot object")
            ok = lyr.addFeature(f)
            if ok:
                lyr.endEditCommand()
                self.saved.append((lyr.id(), f.id(), True))     # temporary negative id until the edits are saved
            else:
                lyr.destroyEditCommand()
        else:
            ok, out = lyr.dataProvider().addFeatures([f])
            if ok:
                self.saved.append((lyr.id(), out[0].id(), False))
                lyr.updateExtents()
        if not ok:
            return self._warn("Could not write to '{}' (read-only?). The object is still on the map.".format(
                lyr.name()))
        lyr.triggerRepaint()
        self.status.setText("Saved to '{}'{} ({} objects, {:.0f} m²). Click the next object.".format(
            lyr.name(), " (temporary – not saved to disk)" if lyr.providerType() == "memory" else "",
            lyr.featureCount(), area))
        return True

    def _undo_saved(self):
        while self.saved:
            lid, fid, as_edit = self.saved.pop()
            lyr = QgsProject.instance().mapLayer(lid)
            if lyr is None:
                continue
            if as_edit:
                # once the edits are saved (or undone) the temporary id is gone: never delete something else
                if not (lyr.isEditable() and lyr.getFeature(fid).isValid()):
                    self.status.setText("Nothing removed: the edits on '{}' were already saved (or undone), "
                                        "so the last object can no longer be taken back here.".format(lyr.name()))
                    return
                lyr.beginEditCommand("Segmentot undo")
                if not lyr.deleteFeature(fid):
                    lyr.destroyEditCommand()
                    self.status.setText("Could not remove the last object from '{}'.".format(lyr.name()))
                    return
                lyr.endEditCommand()
            else:
                lyr.dataProvider().deleteFeatures([fid])
                lyr.updateExtents()
            lyr.triggerRepaint()
            self.status.setText("Removed the last saved object from '{}'.".format(lyr.name()))
            return
        self.status.setText("Nothing to undo.")
