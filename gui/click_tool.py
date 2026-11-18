"""Segmentot map tool.

Left-click = include · Right-click = exclude · Enter = save object · Esc = clear
Backspace = undo last point (or the last saved object when no points)
(Space is not used: QGIS keeps it for panning the map.)
C = next candidate outline (single click only) · hovering previews what a click would give.
"""
import uuid

from qgis.core import QgsApplication, QgsGeometry, QgsRectangle, QgsTask, QgsWkbTypes
from qgis.gui import QgsMapTool, QgsRubberBand, QgsVertexMarker
from qgis.PyQt.QtCore import Qt, QTimer, pyqtSignal
from qgis.PyQt.QtGui import QColor

from ..core import engine

WINDOW_PX = 1024          # SAM2's native input size
INNER = 0.15              # a click this close to a window edge opens a new window centred on it
POLY = QgsWkbTypes.GeometryType.PolygonGeometry
# keep the many tiny hover requests out of the QGIS task bar (Hidden: QGIS 3.26+)
_TASK_FLAGS = getattr(QgsTask.Flag, "Hidden", QgsTask.Flag.CanCancel)


def window_res(grid, upp):
    """Map units per window pixel at canvas zoom upp: the zoom's, or the whole raster pixels closest to it."""
    return upp if grid is None else grid[0] * max(1, round(upp / grid[0]))


def window_at(pt, grid, upp):
    """(rectangle, map units per pixel) of the WINDOW_PX window centred on pt.

    grid = (raster pixel size, origin x, origin y): the window then starts on a raster pixel edge
    and each window pixel is k whole raster pixels, so outlines fall on raster pixel edges."""
    res = window_res(grid, upp)
    if grid is None:
        x0, y1 = pt.x() - WINDOW_PX * res / 2.0, pt.y() + WINDOW_PX * res / 2.0
    else:
        native, ox, oy = grid
        n = WINDOW_PX * round(res / native)        # window size in raster pixels
        x0 = ox + round((pt.x() - ox) / native - n / 2) * native
        y1 = oy - round((oy - pt.y()) / native - n / 2) * native
    return QgsRectangle(x0, y1 - WINDOW_PX * res, x0 + WINDOW_PX * res, y1), res


class SegmentotTool(QgsMapTool):
    undoSaved = pyqtSignal()
    status = pyqtSignal(str)

    def __init__(self, canvas, settings, save):
        """settings(): dict with layer, grid, outline, hover, refine (see plugin.py).
        save(geometry in canvas CRS, score) -> False when the object could not be written."""
        super().__init__(canvas)
        self.canvas = canvas
        self.settings = settings
        self._save = save
        self.preview = self._band(QColor(255, 196, 0, 80), QColor(255, 170, 0), 2, Qt.PenStyle.SolidLine)
        self.hover_band = self._band(QColor(0, 200, 255, 35), QColor(0, 200, 255), 1, Qt.PenStyle.DashLine)
        self.frame = self._band(QColor(0, 0, 0, 0), QColor(255, 255, 255, 110), 1, Qt.PenStyle.DotLine)
        self.hover_timer = QTimer()
        self.hover_timer.setSingleShot(True)
        self.hover_timer.setInterval(220)
        self.hover_timer.timeout.connect(self._hover_fire)
        self.hover_pt = None
        self.markers = []
        self.busy = False
        self.queued = None            # latest request waiting while one is in flight or the image renders
        self.seq = 0                  # bumps on every click/move/reset; stale results are dropped
        self.notice = None            # last engine notice shown (e.g. it fell back to the CPU)
        self.window = self.res = None
        self.image_id = self.image = self._render = None
        self._reset_object()
        self.setCursor(Qt.CursorShape.CrossCursor)
        canvas.scaleChanged.connect(self._zoomed)

    def _band(self, fill, stroke, width, style):
        b = QgsRubberBand(self.canvas, POLY)
        b.setFillColor(fill)
        b.setStrokeColor(stroke)
        b.setWidth(width)
        b.setLineStyle(style)
        return b

    # ------------------------------------------------------------ state
    def _reset_object(self):
        self.seq = getattr(self, "seq", 0) + 1
        self.points, self.labels = [], []
        self.candidates, self.cand_idx = [], 0
        self.raw = self.geom = None
        self.score = 0.0
        self.pending = False          # a click's outline for the current seq has not arrived yet
        self.preview.reset(POLY)
        for m in self.markers:
            self.canvas.scene().removeItem(m)
        self.markers = []

    def _forget_window(self):
        if self._render is not None:
            self._render.cancelWithoutBlocking()
        self.window = self.image_id = self.image = self._render = None
        self.frame.reset(POLY)

    def _open_window(self, pt):
        s = self.settings()
        layer = s["layer"]
        if layer is None:
            self.status.emit("Choose an imagery layer in the Segmentot panel first.")
            return False
        self._forget_window()
        self.window, self.res = window_at(pt, s["grid"], self.canvas.mapUnitsPerPixel())
        self.image_id = uuid.uuid4().hex
        self.frame.setToGeometry(QgsGeometry.fromRect(self.window), None)
        w = self.window
        self._render = engine.render_window(layer, w, WINDOW_PX, self._crs(),
                                            lambda img: self._rendered(w, img))
        return True

    def _rendered(self, window, img):
        if window is not self.window:              # replaced or forgotten while rendering
            return
        self._render, self.image = None, img
        self._flush()

    def _zoomed(self, _scale):
        """A window made for another zoom would show the AI a stale image: drop it between objects."""
        if self.window is not None and not self.points and \
                window_res(self.settings()["grid"], self.canvas.mapUnitsPerPixel()) != self.res:
            self._forget_window()

    def _inside(self, pt, margin=0.0):
        if self.window is None:
            return False
        w = self.window
        return w.buffered(-margin * w.width()).contains(pt) if margin else w.contains(pt)

    def _to_px(self, pt):
        w = self.window
        return [(pt.x() - w.xMinimum()) / w.width() * WINDOW_PX,
                (w.yMaximum() - pt.y()) / w.height() * WINDOW_PX]

    # ------------------------------------------------------------ mouse / keys
    def canvasMoveEvent(self, e):
        if self.points or not self.settings()["hover"]:
            return
        self.seq += 1                              # the preview for the old cursor position is stale
        self.hover_pt = self.toMapCoordinates(e.pos())
        self.hover_band.reset(POLY)
        self.hover_timer.start()

    def _hover_fire(self):
        pt = self.hover_pt
        if pt is None or self.points:
            return
        if not self._inside(pt, INNER):
            if self.busy:                          # a new window would make the next click wait behind it
                self.hover_timer.start()
                return
            layer = self.settings()["layer"]
            # online layers render slowly: only open windows on hover for local files
            if layer is None or layer.providerType() != "gdal" or not self._open_window(pt):
                return
        self._request({"points": [self._to_px(pt)], "labels": [1]}, hover=True)

    def canvasReleaseEvent(self, e):
        self.hover_timer.stop()
        self.hover_band.reset(POLY)
        pt = self.toMapCoordinates(e.pos())
        positive = e.button() == Qt.MouseButton.LeftButton
        if e.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            return
        if self.points and not self._inside(pt):
            if not positive:
                self.status.emit("Exclude points must be inside the dotted frame.")
                return
            if not self.accept():                  # clicked another object: save this one first
                return
        if not self.points:
            if not positive:
                self.status.emit("Start with a left-click on the object.")
                return
            if not self._inside(pt, INNER) and not self._open_window(pt):
                return
        self.points.append(self._to_px(pt))
        self.labels.append(1 if positive else 0)
        m = QgsVertexMarker(self.canvas)
        m.setCenter(pt)
        m.setColor(QColor(40, 200, 90) if positive else QColor(230, 50, 50))
        m.setIconType(QgsVertexMarker.IconType.ICON_CIRCLE if positive else QgsVertexMarker.IconType.ICON_X)
        m.setIconSize(12)
        m.setPenWidth(3)
        self.markers.append(m)
        self.seq += 1
        self._request({"points": list(self.points), "labels": list(self.labels)})

    def keyPressEvent(self, e):
        k = e.key()
        if k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.accept()
        elif k == Qt.Key.Key_Escape:
            self.clear()
        elif k in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            if self.points or not e.isAutoRepeat():    # holding the key must not delete saved objects
                self.undo()
        elif k == Qt.Key.Key_C:
            self.next_candidate()
        else:
            e.ignore()
            return
        e.accept()

    # ------------------------------------------------------------ actions
    def clear(self):
        self._reset_object()
        self.status.emit("Cleared.")

    def undo(self):
        if not self.points:
            self.undoSaved.emit()
            return
        self.points.pop()
        self.labels.pop()
        self.canvas.scene().removeItem(self.markers.pop())
        self.seq += 1
        if self.points:
            self._request({"points": list(self.points), "labels": list(self.labels)})
        else:
            self._reset_object()
            self.status.emit("All points removed.")

    def accept(self):
        """Save the current object. False when it is kept: still segmenting, or writing failed."""
        if self.pending:
            self.status.emit("Still segmenting…")
            return False
        if self.geom is not None and not self.geom.isEmpty():
            if not self._save(QgsGeometry(self.geom), self.score):
                return False                       # the plugin shows why; keep the object to retry
        elif self.points:
            self.status.emit("Nothing to save yet.")
        self._reset_object()
        return True

    def next_candidate(self):
        if len(self.candidates) < 2:
            self.status.emit("Only one outline here (alternatives exist after a single click).")
            return
        self.cand_idx = (self.cand_idx + 1) % len(self.candidates)
        self._use_candidate()

    def refresh(self):
        """Re-apply refine options to the current object (called when the panel changes)."""
        if self.raw is not None:
            self._apply_refine()

    def resend(self):
        """Ask again for the current points (the Outline style changed)."""
        if self.points:
            self.seq += 1
            self._request({"points": list(self.points), "labels": list(self.labels)})

    def deactivate(self):
        try:
            self.hover_timer.stop()
            self.hover_band.reset(POLY)
            self._reset_object()
            self._forget_window()
        except RuntimeError:          # QGIS is closing and Qt already deleted the canvas items
            pass
        super().deactivate()

    def new_imagery(self):
        """Imagery or detail changed: the cached window no longer matches."""
        self._reset_object()
        self._forget_window()

    # ------------------------------------------------------------ engine
    def _request(self, req, hover=False):
        s = self.settings()
        payload = dict(req, image_id=self.image_id, outline=s["outline"], min_area_px=4)
        if not hover:
            self.pending = True
            self.status.emit("Segmenting… (the first one loads the model, ~10 s)")
        self.queued = (payload, self.window, self.res, self.seq, hover)
        self._flush()

    def _flush(self):
        """Send the waiting request once nothing is in flight and its window's image is ready."""
        job = self.queued
        if job is None or self.busy or self.image is None:
            return
        self.queued = None
        if job[1] is self.window and job[3] == self.seq:     # else its result would be dropped anyway
            self._send(job)

    def _send(self, job):
        payload, window, res, seq, hover = job
        image = self.image
        self.busy = True

        def work(task):
            r = engine.call("/predict", payload)
            if r.get("need_image"):
                r = engine.call("/predict", dict(payload, image=engine.encode_png(image)))
            return r

        def done(exc, result=None):
            self.busy = False
            if not hover and seq == self.seq:
                self.pending = False
            self._flush()
            if exc is not None:
                self.status.emit(str(exc))
                return
            if hover:
                if not self.points and seq == self.seq:
                    self._show_hover(result, window, res)
            elif seq == self.seq:
                self._show(result, window, res)
            if result.get("notice") and result["notice"] != self.notice:
                self.notice = result["notice"]
                self.status.emit(self.notice)

        self._task = QgsTask.fromFunction("Segmentot", work, on_finished=done,
                                          flags=_TASK_FLAGS)
        QgsApplication.taskManager().addTask(self._task)

    def _crs(self):
        return self.canvas.mapSettings().destinationCrs()

    def _show_hover(self, result, window, res):
        c = result.get("candidates") or []
        if not c:
            return
        g = engine.refine(engine.pixel_polys_to_geometry(c[0]["polygons"], window, WINDOW_PX),
                          self.settings()["refine"], res, self._crs())
        if g is not None and not g.isEmpty():
            self.hover_band.setToGeometry(g, None)

    def _show(self, result, window, res):
        self.candidates = [(engine.pixel_polys_to_geometry(c["polygons"], window, WINDOW_PX), c["score"])
                           for c in result.get("candidates") or []]
        self.candidates = [c for c in self.candidates if not c[0].isEmpty()]
        self.cand_idx = 0
        self.secs = result.get("seconds", 0)
        if not self.candidates:
            self.raw = self.geom = None
            self.preview.reset(POLY)
            self.status.emit(result.get("error") or "Nothing found there. Try another point.")
            return
        self._use_candidate()

    def _use_candidate(self):
        self.raw, self.score = self.candidates[self.cand_idx]
        self._apply_refine()

    def _apply_refine(self):
        self.geom = engine.refine(self.raw, self.settings()["refine"], self.res, self._crs())
        self.preview.reset(POLY)
        if self.geom is None or self.geom.isEmpty():
            self.status.emit("Refine options removed the whole object (min area too big?).")
            return
        self.preview.setToGeometry(self.geom, None)
        alt = "  ·  outline {}/{} (C = next)".format(self.cand_idx + 1, len(self.candidates)) \
            if len(self.candidates) > 1 else ""
        cut = "" if self.window.buffered(-self.res).contains(self.raw.boundingBox()) else \
            "\nThe object reaches the dotted frame and may be cut: zoom out and click it again."
        self.status.emit("Score {:.2f}  ·  {:.0f} m²{}  ({:.2f} s){}".format(
            self.score, engine.area_m2(self.geom, self._crs()), alt, self.secs, cut))
