"""End-to-end test: headless QGIS -> the plugin's panel and map tool -> the real engine -> the real SAM2.1 model.

It builds a synthetic GeoTIFF with objects at known positions, clicks them through the real map tool,
and compares the saved polygons with the truth. Needs QGIS's Python (see README.md here) and the engine;
if the engine is not installed it runs the plugin's own installer first (a download of roughly 1-5 GB).
"""
import os
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request

import numpy as np
from osgeo import gdal, osr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.environ.get("SEGMENTOT_TEST_PARENT") or os.path.dirname(REPO))
PKG = os.path.basename(REPO)

from qgis.core import (  # noqa: E402
    QgsApplication, QgsCoordinateReferenceSystem, QgsGeometry, QgsPointXY,
    QgsProject, QgsRasterLayer, QgsRectangle, QgsVectorLayer,
)
from qgis.gui import QgsMapCanvas, QgsMapMouseEvent, QgsMessageBar  # noqa: E402
from qgis.PyQt.QtCore import QEvent, QPoint, QSettings, Qt  # noqa: E402
from qgis.PyQt.QtGui import QKeyEvent  # noqa: E402
from qgis.PyQt.QtWidgets import QMainWindow  # noqa: E402

RES, OX, OY, N = 0.5, 700000.0, 9300000.0, 3000          # metres per pixel, origin, image size
EPSG = 32748                                              # UTM 48S (Java)


# ------------------------------------------------------------------ synthetic imagery with known truth
def px_poly(pts, holes=()):
    rings = [[QgsPointXY(OX + x * RES, OY - y * RES) for x, y in pts]]
    for h in holes:
        rings.append([QgsPointXY(OX + x * RES, OY - y * RES) for x, y in h])
    return QgsGeometry.fromPolygonXY(rings)


def px_pt(x, y):
    return QgsPointXY(OX + x * RES, OY - y * RES)


def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def rot_rect(cx, cy, w, h, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    pts = [(cx + dx * c - dy * s, cy + dx * s + dy * c)
           for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]
    return pts + [pts[0]]


def make_image(path):
    rng = np.random.default_rng(7)
    coarse = rng.normal(0, 1, (N // 50, N // 50))
    tex = np.kron(coarse, np.ones((50, 50)))                       # blocky low-frequency texture
    img = np.stack([70 + 12 * tex, 110 + 14 * tex, 60 + 10 * tex], axis=2) + rng.normal(0, 5, (N, N, 3))
    yy, xx = np.mgrid[0:N, 0:N]
    truth = {}

    def paint(mask, colour, noise=4.0):
        img[mask] = np.array(colour, dtype=float) + (rng.normal(0, noise, (int(mask.sum()), 3)) if noise else 0)

    paint((xx >= 400) & (xx < 480) & (yy >= 400) & (yy < 450), (200, 60, 50))
    truth["building"] = px_poly(rect(400, 400, 480, 450))
    cx, cy, w, h, deg = 800.0, 420.0, 90.0, 50.0, 30.0
    a = np.radians(deg)
    dx, dy = xx + 0.5 - cx, yy + 0.5 - cy
    u, v = dx * np.cos(a) + dy * np.sin(a), -dx * np.sin(a) + dy * np.cos(a)
    paint((np.abs(u) <= w / 2) & (np.abs(v) <= h / 2), (182, 182, 188))
    truth["rotated"] = px_poly(rot_rect(cx, cy, w, h, deg))
    paint((xx >= 650) & (xx < 670) & (yy >= 700) & (yy < 720), (240, 240, 240))
    truth["small"] = px_poly(rect(650, 700, 670, 720))
    lake = ((xx + 0.5 - 500) / 180.0) ** 2 + ((yy + 0.5 - 1100) / 120.0) ** 2 <= 1
    paint(lake, (0, 0, 0), noise=0)                                # pure black: must not count as "no data"
    t = np.linspace(0, 2 * np.pi, 361)
    truth["lake"] = px_poly([(500 + 180 * np.cos(k), 1100 + 120 * np.sin(k)) for k in t])
    paint((xx >= 1400) & (xx < 2900) & (yy >= 300) & (yy < 1500), (190, 170, 90))
    truth["field"] = px_poly(rect(1400, 300, 2900, 1500))          # 1500 x 1200 px: bigger than one window
    block = (xx >= 850) & (xx < 1150) & (yy >= 1850) & (yy < 2150)
    img[block] += rng.normal(0, 35, (int(block.sum()), 3))         # speckle, like real imagery
    paint((xx >= 990) & (xx < 1010) & (yy >= 1990) & (yy < 2010), (240, 240, 240), noise=35.0)
    truth["speck"] = px_poly(rect(990, 1990, 1010, 2010))
    # courtyard building: 200 x 200 outer with an 80 x 80 inner courtyard
    court_out = (xx >= 1600) & (xx < 1800) & (yy >= 1900) & (yy < 2100)
    court_in = (xx >= 1660) & (xx < 1740) & (yy >= 1960) & (yy < 2040)
    paint(court_out & ~court_in, (185, 95, 75), noise=5.0)
    truth["courtyard"] = px_poly(rect(1600, 1900, 1800, 2100), [rect(1660, 1960, 1740, 2040)])
    # hierarchical compound: pad -> building -> bright skylight unit
    paint((xx >= 2100) & (xx < 2400) & (yy >= 1880) & (yy < 2120), (150, 150, 145), noise=5.0)
    paint((xx >= 2140) & (xx < 2360) & (yy >= 1920) & (yy < 2080), (190, 90, 60), noise=5.0)
    paint((xx >= 2220) & (xx < 2280) & (yy >= 1970) & (yy < 2030), (235, 235, 240), noise=4.0)
    truth["compound_pad"] = px_poly(rect(2100, 1880, 2400, 2120))
    truth["compound_bldg"] = px_poly(rect(2140, 1920, 2360, 2080))
    truth["compound_unit"] = px_poly(rect(2220, 1970, 2280, 2030))
    # off-center auto-recenter pair: anchor opens a window that clips edge_bldg on the left
    paint((xx >= 770) & (xx < 830) & (yy >= 2320) & (yy < 2380), (210, 180, 70), noise=5.0)
    truth["anchor"] = px_poly(rect(770, 2320, 830, 2380))
    paint((xx >= 150) & (xx < 550) & (yy >= 2200) & (yy < 2500), (175, 85, 65), noise=5.0)
    truth["edge_bldg"] = px_poly(rect(150, 2200, 550, 2500))
    img = np.clip(img, 1, 255)
    img[lake] = 0
    ds = gdal.GetDriverByName("GTiff").Create(path, N, N, 3, gdal.GDT_Byte, ["TILED=YES", "COMPRESS=DEFLATE"])
    ds.SetGeoTransform((OX, RES, 0, OY, 0, -RES))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EPSG)
    ds.SetProjection(srs.ExportToWkt())
    for b in range(3):
        ds.GetRasterBand(b + 1).WriteArray(img[..., b].astype(np.uint8))
    ds.BuildOverviews("AVERAGE", [2, 4, 8])
    ds = None
    return truth


def centre(g):
    return g.centroid().asPoint()


def iou(a, b):
    return a.intersection(b).area() / max(a.combine(b).area(), 1e-9)


# ------------------------------------------------------------------ a stand-in for QGIS's iface
class FakeIface:
    def __init__(self, win, canvas):
        self._win, self._canvas, self._bar = win, canvas, QgsMessageBar(win)
        self.bar_messages = []
        push = self._bar.pushWarning

        def record(title, text):
            self.bar_messages.append(text)
            push(title, text)
        self._bar.pushWarning = record

    def mainWindow(self):
        return self._win

    def mapCanvas(self):
        return self._canvas

    def messageBar(self):
        return self._bar

    def addDockWidget(self, area, dock):
        self._win.addDockWidget(area, dock)

    def removeDockWidget(self, dock):
        self._win.removeDockWidget(dock)

    def activeLayer(self):
        return None

    def __getattr__(self, name):          # addToolBarIcon, addPluginToRasterMenu, remove..., actionPan ...
        return lambda *a, **k: None


# ------------------------------------------------------------------ the run
class Run:
    def __init__(self):
        self.results = []

    def check(self, name, ok, detail=""):
        self.results.append((name, bool(ok), detail))
        print("{}  {}{}".format("PASS" if ok else "FAIL", name, "  — " + detail if detail else ""), flush=True)


def main():
    app = QgsApplication([], True)
    app.setOrganizationName("SegmentotE2E")       # keep the test's QSettings away from the real QGIS profile
    app.setApplicationName("SegmentotE2E")
    app.initQgis()
    QSettings().setValue("Segmentot/port", 8793)   # not the port of a QGIS that may be open on this computer
    QSettings().setValue("Segmentot/model", os.environ.get("SEGMENTOT_TEST_MODEL", "base"))     # e.g. edge
    pkg = __import__(PKG)
    runtime = __import__(PKG + ".core.runtime", fromlist=["x"])
    engine = __import__(PKG + ".core.engine", fromlist=["x"])
    plugin_mod = __import__(PKG + ".gui.plugin", fromlist=["x"])
    assert pkg and callable(pkg.classFactory)
    R = Run()
    tmp = tempfile.mkdtemp(prefix="segmentot_e2e_")

    # ---- 0. engine installed (the plugin's own installer; this is the install end-to-end)
    if not runtime.is_ready():
        print("Engine not installed: running the plugin's installer into", runtime.home(), flush=True)
        t0 = time.time()
        last = [0]

        def progress(p):
            if int(p) != last[0]:
                last[0] = int(p)
                print("  [{:3d}%]".format(int(p)), flush=True)
        try:
            runtime.setup(lambda s: print("  " + s, flush=True), progress, lambda: False)
        except Exception as e:
            R.check("install: engine set up by the plugin's installer", False, str(e))
            return finish(R, app)
        print("Install took {:.0f} s".format(time.time() - t0), flush=True)
    R.check("install: engine is ready", runtime.is_ready(), runtime.home())
    st = runtime.load_state()
    want = QSettings().value("Segmentot/model")
    if not runtime.model_downloaded(want):                            # the install leaves the model out
        runtime.download_model(want, lambda s: print("  " + s, flush=True), lambda: False)
    info = st.get("detected", {})
    print("Detected:", runtime.describe(info) if info else "?", "| planned accel:", info.get("accel"),
          "| torch:", st.get("torch"), flush=True)

    # ---- imagery, project, canvas, panel
    tif = os.path.join(tmp, "scene.tif")
    truth = make_image(tif)
    layer = QgsRasterLayer(tif, "scene")
    assert layer.isValid(), "test raster did not load"
    crs = QgsCoordinateReferenceSystem.fromEpsgId(EPSG)
    QgsProject.instance().setCrs(crs)
    QgsProject.instance().addMapLayer(layer)
    win = QMainWindow()
    canvas = QgsMapCanvas()
    win.setCentralWidget(canvas)
    win.resize(1500, 900)
    win.show()
    canvas.setDestinationCrs(crs)
    canvas.setLayers([layer])
    app.processEvents()

    def pump(seconds=0.0):
        t0 = time.time()
        while True:
            app.processEvents()
            if time.time() - t0 >= seconds:
                return
            time.sleep(0.01)

    def view(pt, mupp):
        w, h = canvas.width() * mupp, canvas.height() * mupp
        canvas.setExtent(QgsRectangle(pt.x() - w / 2, pt.y() - h / 2, pt.x() + w / 2, pt.y() + h / 2))
        canvas.refresh()
        pump(0.3)

    iface = FakeIface(win, canvas)
    plugin = plugin_mod.SegmentotPlugin(iface)
    plugin.initGui()
    tool = plugin.tool
    statuses = []
    tool.status.connect(statuses.append)
    plugin._check_engine = lambda: None     # the panel starts the engine when shown (ui_smoke.py checks that);
    plugin.dock.show()                      # here the engine is started further down, after the no-model checks
    plugin.raster.setLayer(layer)

    out = os.path.join(tmp, "objects.gpkg")
    plugin_mod.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (out, ""))    # the dialog's answer
    try:
        plugin._new_gpkg_layer()                                   # the panel's "New file…" button
    except Exception:
        print(traceback.format_exc(), flush=True)
    target = plugin.target.currentLayer()
    R.check("panel: 'New file…' creates the layer and selects it as the target",
            target is not None and target.providerType() == "ogr" and os.path.exists(out),
            "target={}".format(target.name() if target else None))
    if target is None:
        return finish(R, app)

    def wait(pred, timeout, what):
        t0 = time.time()
        while time.time() - t0 < timeout:
            app.processEvents()
            if pred():
                return time.time() - t0
            time.sleep(0.01)
        raise AssertionError("timed out after {} s waiting for {}; last status: {!r}".format(
            timeout, what, statuses[-1] if statuses else ""))

    def mouse(pt, button=Qt.MouseButton.LeftButton, kind=QEvent.Type.MouseButtonRelease):
        p = canvas.getCoordinateTransform().transform(pt)
        return QgsMapMouseEvent(canvas, kind, QPoint(int(round(p.x())), int(round(p.y()))), button)

    def click(pt, button=Qt.MouseButton.LeftButton, timeout=60):
        """Click and wait for the outline; returns (seconds, status line)."""
        n = len(statuses)
        tool.canvasReleaseEvent(mouse(pt, button))
        t = wait(lambda: any(s.startswith("Score") or "found" in s or "No imagery" in s or "problem" in s.lower()
                             or "error" in s.lower() for s in statuses[n:]) and not tool.busy,
                 timeout, "an outline")
        return t, "\n".join(statuses[n:])

    def key(k, autorepeat=False):
        tool.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, k, Qt.KeyboardModifier.NoModifier, "", autorepeat))
        pump(0.05)

    def count():
        return target.dataProvider().featureCount()

    def last_feature():
        feats = sorted(target.getFeatures(), key=lambda f: f.id())
        return feats[-1]

    try:
        # ---- 0b. no model on this computer: the panel shows the setup steps, nothing is downloaded silently
        real_downloaded, real_answers = runtime.model_downloaded, plugin._engine_answers
        runtime.model_downloaded = lambda key: False
        plugin._engine_answers = lambda: False          # also when another QGIS has an engine running
        try:
            plugin._refresh()
            plugin.start_btn.setChecked(True)
            R.check("model: without a model the panel shows step 2 (with the size) and Start does not start",
                    canvas.mapTool() is not tool and not plugin.setup_page.isHidden() and plugin.work_page.isHidden()
                    and "MB" in plugin.model_btn.text() and plugin.model_btn.isEnabled()
                    and plugin.install_btn.isHidden(), plugin.model_btn.text())
            try:
                engine.ensure_engine()
                err = ""
            except engine.EngineError as e:
                err = str(e)
            R.check("model: the engine does not start (or download anything) without the model",
                    "model" in err.lower() and not runtime.engine_running_here(), err)
        finally:
            runtime.model_downloaded, plugin._engine_answers = real_downloaded, real_answers
        plugin._refresh()
        R.check("model: with the model present the panel shows the work screen",
                plugin.setup_page.isHidden() and not plugin.work_page.isHidden())
        R.check("panel: the action buttons are off until Start is pressed, the guide says what to do",
                not any(b.isEnabled() for b in plugin.actions) and "Start" in plugin.status.text(),
                plugin.status.text())

        # ---- 1. engine: GPU first, security
        view(centre(truth["building"]), RES)
        plugin.start_btn.setChecked(True)
        R.check("panel: Start activates the map tool and the action buttons", canvas.mapTool() is tool
                and all(b.isEnabled() for b in plugin.actions) and not plugin.keys.isHidden(), plugin.status.text())
        R.check("engine: kept awake while Start is on", plugin._alive.isActive())
        t0 = time.time()
        health = engine.call("/health", timeout=600, headers={"X-Segmentot": "load"})
        print("Engine start + model load: {:.1f} s | {}".format(time.time() - t0, health), flush=True)
        want = {"cuda": "cuda", "mps": "mps"}.get(info.get("accel"), "cpu")
        R.check("engine: uses the GPU first ({} expected on this computer)".format(want),
                health.get("device") == want and not health.get("notice"),
                "device={} notice={}".format(health.get("device"), health.get("notice")))
        try:
            urllib.request.build_opener(urllib.request.ProxyHandler({})).open(engine.url() + "/health", timeout=5)
            code = 200
        except urllib.error.HTTPError as e:
            code = e.code
        R.check("engine: request without the Segmentot header is refused", code == 403, "HTTP {}".format(code))

        # ---- 2. plain building, default settings (raster pixels + pixel edges), save with a class
        plugin.cls.setText("building")
        pump(0.1)
        rows_at = [plugin.actions[0].y()]           # where the buttons under the guide box sit
        t, s = click(centre(truth["building"]), timeout=180)
        pump(0.1)
        rows_at.append(plugin.actions[0].y())
        v = iou(tool.geom, truth["building"])
        R.check("click: building outline matches the truth", v >= 0.95, "IoU {:.3f}, {:.2f} s".format(v, t))
        key(Qt.Key.Key_Return)
        R.check("save: Enter writes one feature to the GeoPackage", count() == 1, "{} features".format(count()))
        pump(0.1)
        rows_at.append(plugin.actions[0].y())
        R.check("panel: the guide text changing (hint, score, saved) does not move the panel below it",
                len(set(rows_at)) == 1, "y = {} at dock width {}".format(rows_at, plugin.dock.width()))
        f = last_feature()
        R.check("save: attributes written (class, area)", f["class"] == "building"
                and abs(f["area_m2"] - 1000) / 1000 < 0.05, "class={} area_m2={}".format(f["class"], f["area_m2"]))
        R.check("save: stored polygon matches the truth", iou(f.geometry(), truth["building"]) >= 0.95,
                "IoU {:.3f}".format(iou(f.geometry(), truth["building"])))

        # ---- 3. pure-black lake must come back whole (finding 2)
        view(centre(truth["lake"]), RES)
        t, s = click(centre(truth["lake"]))
        g = tool.geom
        holes = sum(len(p) - 1 for p in (g.asMultiPolygon() if g.isMultipart() else [g.asPolygon()]))
        v = iou(g, truth["lake"])
        R.check("accuracy: pure-black lake is one whole polygon", v >= 0.95 and holes == 0,
                "IoU {:.3f}, {} holes, {:.2f} s".format(v, holes, t))
        key(Qt.Key.Key_Escape)

        # ---- 4. Smooth outline keeps the area of a small object (finding 9)
        view(centre(truth["small"]), RES)
        plugin.outline.setCurrentIndex(1)
        tool.new_imagery()                 # a new window centred on the object, not a kept one where it is off-centre
        pump(0.2)
        t, s = click(centre(truth["small"]))
        a, ta = tool.geom.area(), truth["small"].area()
        R.check("accuracy: Smooth outline keeps a 20x20 px object's area", abs(a - ta) / ta < 0.08,
                "{:.1f} m² vs {:.1f} m² ({:+.1f}%)".format(a, ta, 100 * (a - ta) / ta))
        # switching the outline style must keep the object and redraw it (finding 19)
        n = len(statuses)
        plugin.outline.setCurrentIndex(0)
        wait(lambda: any(x.startswith("Score") for x in statuses[n:]) and not tool.busy, 30, "outline re-sent")
        R.check("panel: changing Outline keeps the current object", bool(tool.points) and tool.geom is not None
                and not tool.geom.isEmpty(), "{} point(s)".format(len(tool.points)))
        key(Qt.Key.Key_Escape)

        # ---- 5. Square corners on the default pixel-edge outline (finding 10)
        view(centre(truth["rotated"]), RES)
        t, s = click(centre(truth["rotated"]))
        raw_v = tool.geom.constGet().nCoordinates()
        plugin.square.setChecked(True)
        pump(0.2)
        sq = tool.geom
        v = iou(sq, truth["rotated"])
        R.check("refine: Square corners squares a rotated building", sq.constGet().nCoordinates() <= 6 and v >= 0.92,
                "{} -> {} vertices, IoU {:.3f}".format(raw_v, sq.constGet().nCoordinates(), v))
        plugin.square.setChecked(False)
        plugin.smooth.setChecked(True)
        pump(0.2)
        a, ta = tool.geom.area(), truth["rotated"].area()
        R.check("refine: Smooth curves keeps the area", abs(a - ta) / ta < 0.06,
                "{:.1f} vs {:.1f} m² ({:+.1f}%)".format(a, ta, 100 * (a - ta) / ta))
        plugin.smooth.setChecked(False)
        key(Qt.Key.Key_Escape)

        # ---- 5b. Square corners on an axis-aligned building keeps full IoU
        view(centre(truth["building"]), RES)
        t, s = click(centre(truth["building"]))
        plugin.square.setChecked(True)
        pump(0.2)
        sq = tool.geom
        v = iou(sq, truth["building"])
        R.check("refine: Square corners keeps an axis-aligned building",
                sq.constGet().nCoordinates() <= 6 and v >= 0.97,
                "{} vertices, IoU {:.3f}".format(sq.constGet().nCoordinates(), v))
        plugin.square.setChecked(False)
        key(Qt.Key.Key_Escape)

        # ---- 5c. Courtyard building: roof click + right-click inside courtyard hole
        view(px_pt(1700, 2000), RES)
        click(px_pt(1630, 1930))
        click(px_pt(1700, 2000), Qt.MouseButton.RightButton)
        g = tool.geom
        holes = sum(len(p) - 1 for p in (g.asMultiPolygon() if g.isMultipart() else [g.asPolygon()]))
        v = iou(g, truth["courtyard"])
        R.check("accuracy: courtyard building keeps its inner courtyard hole",
                v >= 0.95 and holes >= 1, "IoU {:.3f}, {} hole(s)".format(v, holes))
        key(Qt.Key.Key_Escape)

        # ---- 5d. Hierarchical compound: C selects building, right-click on pad keeps granularity via mask_id
        view(px_pt(2250, 2000), RES)
        click(px_pt(2250, 2000))
        best_v, best_idx = -1.0, 0
        for _ in range(len(tool.candidates)):
            cv = iou(tool.geom, truth["compound_bldg"])
            if cv > best_v:
                best_v, best_idx = cv, tool.cand_idx
            key(Qt.Key.Key_C)
        while tool.cand_idx != best_idx:
            key(Qt.Key.Key_C)
        click(px_pt(2115, 2000), Qt.MouseButton.RightButton)
        v = iou(tool.geom, truth["compound_bldg"])
        R.check("accuracy: C candidate choice survives a refine click",
                len(tool.candidates) >= 1 and best_v >= 0.8 and v >= 0.92,
                "chosen cand IoU {:.3f} -> after exclude click {:.3f}".format(best_v, v))
        key(Qt.Key.Key_Escape)

        # ---- 5e. Off-center click inside a cached window auto-recenters instead of clipping the object
        view(px_pt(650, 2350), RES)
        tool.new_imagery()
        click(centre(truth["anchor"]))
        cached_id = tool.image_id
        key(Qt.Key.Key_Escape)
        t, s = click(px_pt(480, 2350))          # inside INNER of cached window, but edge_bldg hits its left edge
        v = iou(tool.geom, truth["edge_bldg"])
        R.check("accuracy: off-center clipped object in a cached window auto-recenters",
                tool.image_id != cached_id and v >= 0.95 and "frame" not in s.lower(),
                "recentered={}, IoU {:.3f}, {:.2f} s".format(tool.image_id != cached_id, v, t))
        key(Qt.Key.Key_Escape)

        # ---- 6. exclude click, Enter too early, then save (findings 6, 7)
        view(centre(truth["building"]), RES)
        c = centre(truth["building"])
        t, s = click(c)
        t, s = click(QgsPointXY(c.x() + 45, c.y()), Qt.MouseButton.RightButton)     # background, inside the frame
        v = iou(tool.geom, truth["building"])
        R.check("click: an exclude click keeps the building outline", v >= 0.93, "IoU {:.3f}".format(v))
        key(Qt.Key.Key_Escape)
        before = count()
        n = len(statuses)
        tool.canvasReleaseEvent(mouse(c))
        waiting = canvas.cursor().shape()
        tool.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier))
        R.check("save: Enter before the outline arrives saves nothing", count() == before and bool(tool.points),
                "status: {!r}".format(statuses[-1]))
        wait(lambda: any(x.startswith("Score") for x in statuses[n:]) and not tool.busy, 60, "the outline")
        R.check("click: the map shows a busy cursor until the outline arrives",
                waiting == Qt.CursorShape.BusyCursor and canvas.cursor().shape() == Qt.CursorShape.CrossCursor)
        key(Qt.Key.Key_Return)
        R.check("save: Enter after the outline arrives saves", count() == before + 1, "{} features".format(count()))

        # ---- 7. Backspace auto-repeat must not delete saved objects (finding 1)
        before = count()
        for _ in range(15):
            key(Qt.Key.Key_Backspace, autorepeat=True)
        R.check("undo: holding Backspace deletes nothing", count() == before, "{} -> {}".format(before, count()))
        key(Qt.Key.Key_Backspace)
        R.check("undo: one Backspace press removes the last saved object", count() == before - 1,
                "{} -> {}".format(before, count()))

        # ---- 7b. a failed save keeps the object and shows a warning (finding 7)
        view(centre(truth["building"]), RES)
        t, s = click(centre(truth["building"]))
        before, real_save = count(), tool._save
        tool._save = lambda g, sc: plugin._warn("Could not write (simulated by the test).")
        key(Qt.Key.Key_Return)
        R.check("save: a failed save keeps the object and warns in the message bar", count() == before
                and bool(tool.points) and tool.geom is not None and any("simulated" in m for m in iface.bar_messages),
                "{} point(s), bar: {}".format(len(tool.points), iface.bar_messages[-1:] if iface.bar_messages else []))
        tool._save = real_save
        key(Qt.Key.Key_Return)
        R.check("save: retrying after the failure saves the kept object", count() == before + 1 and not tool.points)

        # ---- 7c. Class box is disabled for a layer without a 'class' field (finding 8)
        plain = QgsVectorLayer("Polygon?crs=EPSG:{}".format(EPSG), "no class field", "memory")
        QgsProject.instance().addMapLayer(plain)
        plugin.target.setLayer(plain)
        off = not plugin.cls.isEnabled()
        plugin.target.setLayer(target)
        R.check("panel: Class is disabled for a layer without a 'class' field", off and plugin.cls.isEnabled())

        # ---- 8. big field: frame warning, then zooming out widens the window (finding 4)
        fc = centre(truth["field"])
        edge = QgsPointXY(OX + 1500 * RES, OY - 900 * RES)       # 100 px inside the field's left edge
        view(edge, RES)
        t, s = click(edge)
        R.check("accuracy: an object cut by the frame warns", "frame" in s.lower(), "status: {!r}".format(s[-160:]))
        key(Qt.Key.Key_Escape)
        view(fc, RES * 4)
        t, s = click(fc)
        v = iou(tool.geom, truth["field"])
        R.check("accuracy: zooming out captures the whole field in raster-pixel mode",
                abs(tool.res - RES * 4) < 1e-9 and v >= 0.95 and "frame" not in s.lower(),
                "window {:.2f} m/px, IoU {:.3f}, {:.2f} s".format(tool.res, v, t))
        # an exclude click in the middle of an even object says little: it must not wreck the outline
        a0 = tool.geom.area()
        click(QgsPointXY(fc.x() + 150, fc.y() + 100), Qt.MouseButton.RightButton)
        R.check("click: an exclude click inside a big even object does not collapse its outline",
                tool.geom.area() >= 0.5 * a0, "{:.0f} -> {:.0f} m²".format(a0, tool.geom.area()))
        key(Qt.Key.Key_Escape)

        # ---- 9. map-zoom mode follows the zoom (finding 3)
        plugin.detail.setCurrentIndex(1)
        pump(0.2)
        b = centre(truth["building"])
        view(b, 1.0)
        t, s = click(b)
        coarse = tool.res
        key(Qt.Key.Key_Escape)
        view(b, 0.25)
        t, s = click(b)
        v = iou(tool.geom, truth["building"])
        R.check("accuracy: after zooming in, the window uses the new zoom", abs(coarse - 1.0) < 0.01
                and abs(tool.res - canvas.mapUnitsPerPixel()) < 1e-9 and abs(tool.res - 0.25) < 0.01 and v >= 0.95,
                "{:.2f} -> {:.2f} m/px, IoU {:.3f}".format(coarse, tool.res, v))
        key(Qt.Key.Key_Escape)
        plugin.detail.setCurrentIndex(0)
        pump(0.2)

        # ---- 10. hover preview
        view(centre(truth["rotated"]), RES)
        tool.canvasMoveEvent(mouse(centre(truth["rotated"]), Qt.MouseButton.NoButton, QEvent.Type.MouseMove))
        try:
            wait(lambda: not tool.hover_band.asGeometry().isEmpty(), 30, "the hover preview")
            v = iou(tool.hover_band.asGeometry(), truth["rotated"])
        except AssertionError as e:
            v = 0.0
            print(e)
        R.check("hover: preview outline appears under the cursor", v >= 0.9, "IoU {:.3f}".format(v))

        # ---- 10c. a small object in speckled imagery: the hint, then Map zoom and zooming in sharpens it
        sp = centre(truth["speck"])
        view(sp, RES)
        t, s = click(sp)
        v1 = iou(tool.geom, truth["speck"])
        key(Qt.Key.Key_Escape)
        plugin.detail.setCurrentIndex(1)
        view(sp, RES / 4)
        click(sp)
        v2 = iou(tool.geom, truth["speck"])
        R.check("accuracy: a small object gets the Map zoom hint, and zooming in there sharpens it",
                "Small object" in s and "Map zoom" in s and v1 >= 0.7 and v2 >= 0.9,
                "IoU {:.3f} at raster pixels, {:.3f} zoomed in 4x".format(v1, v2))
        key(Qt.Key.Key_Escape)
        plugin.detail.setCurrentIndex(0)
        pump(0.2)

        # ---- 11. speed: a new window, a click inside it, and going back to an earlier window
        view(b, RES)
        tool.new_imagery()
        t1, _ = click(b)
        first = tool.image_id
        key(Qt.Key.Key_Escape)
        t2, _ = click(QgsPointXY(b.x() + 5, b.y() + 3))
        key(Qt.Key.Key_Escape)
        click(centre(truth["rotated"]))             # 200 m away: outside the middle of the first window
        other = tool.image_id
        key(Qt.Key.Key_Escape)
        t3, _ = click(b)
        print("Click: new window {:.2f} s, same window {:.2f} s, back to an earlier window {:.2f} s".format(
            t1, t2, t3), flush=True)
        R.check("speed: a click that opens a new window answers within 10 s", t1 < 10.0, "{:.2f} s".format(t1))
        R.check("speed: a click in an existing window answers within 3 s", t2 < 3.0, "{:.2f} s".format(t2))
        R.check("speed: going back to an earlier window reuses it", other != first and tool.image_id == first
                and t3 < 3.0, "{:.2f} s".format(t3))
        key(Qt.Key.Key_Escape)
        layer.emitStyleChanged()
        R.check("imagery: a style change drops the kept windows", tool.window is None and not tool.views)
    except Exception:
        R.check("run completed without an exception", False, traceback.format_exc()[-1500:])
    finally:
        try:
            plugin.unload()
            pump(0.5)
            R.check("unload: plugin unloads and stops its engine", not runtime.engine_running_here()
                    and not plugin._alive.isActive())
        except Exception:
            R.check("unload: plugin unloads and stops its engine", False, traceback.format_exc()[-800:])
    if iface.bar_messages:
        print("Message bar:", iface.bar_messages, flush=True)
    return finish(R, app)


def finish(R, app):
    bad = [r for r in R.results if not r[1]]
    print("\n{} checks, {} failed".format(len(R.results), len(bad)), flush=True)
    sys.stdout.flush()
    os._exit(1 if bad else 0)      # skip Qt teardown: QGIS objects die in an undefined order headless


if __name__ == "__main__":
    main()
