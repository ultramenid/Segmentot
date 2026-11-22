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
KEEP = 12                 # finished windows kept for reuse (fits inside the engine's warm cache)
HOVER_WARM_MS = 45        # debounce when hovering inside an already-rendered window
HOVER_COLD_MS = 220       # debounce when hovering would open a new window
PREWARM_MS = 150          # idle delay after Start / pan / zoom before pre-warming the viewport window
POLY = QgsWkbTypes.GeometryType.PolygonGeometry
# keep the many tiny hover requests out of the QGIS task bar (Hidden: QGIS 3.26+)
_HIDDEN_FLAG = getattr(QgsTask.Flag, "Hidden", None)
_TASK_FLAGS = (QgsTask.Flag.CanCancel | _HIDDEN_FLAG) if _HIDDEN_FLAG is not None else QgsTask.Flag.CanCancel
_MOD_MASK = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier | Qt.KeyboardModifier.AltModifier


def window_res(grid, upp):
    """Map units per window pixel at canvas zoom upp: the zoom's, or the whole raster pixels closest to it."""
    return upp if grid is None else grid[0] * max(1, round(upp / grid[0]))


def window_at(pt, grid, upp, window_px=WINDOW_PX):
    """(rectangle, map units per pixel) of the window_px window centred on pt.

    grid = (raster pixel size, origin x, origin y): the window then starts on a raster pixel edge
    and each window pixel is k whole raster pixels, so outlines fall on raster pixel edges."""
    res = window_res(grid, upp)
    if grid is None:
        x0, y1 = pt.x() - window_px * res / 2.0, pt.y() + window_px * res / 2.0
    else:
        native, ox, oy = grid
        n = window_px * round(res / native)        # window size in raster pixels
        x0 = ox + round((pt.x() - ox) / native - n / 2) * native
        y1 = oy - round((oy - pt.y()) / native - n / 2) * native
    return QgsRectangle(x0, y1 - window_px * res, x0 + window_px * res, y1), res


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
        self._alive = True
        self._active = False
        self._device = None
        self._tasks = set()
        self._sent_ids = set()
        self._hover_enabled = True
        self._hover_geom = None
        self._inflight_task = None
        self._inflight_hover = False
        self._inflight_image_id = None
        self._prewarm_window = self._prewarm_res = self._prewarm_id = self._prewarm_render = None
        self._prewarm_task = self._prewarm_encoding_id = None
        self.preview = self._band(QColor(255, 196, 0, 80), QColor(255, 170, 0), 2, Qt.PenStyle.SolidLine)
        self.hover_band = self._band(QColor(0, 200, 255, 35), QColor(0, 200, 255), 1, Qt.PenStyle.DashLine)
        self.frame = self._band(QColor(0, 0, 0, 0), QColor(255, 255, 255, 110), 1, Qt.PenStyle.DotLine)
        self.hover_timer = QTimer()
        self.hover_timer.setSingleShot(True)
        self.hover_timer.setInterval(HOVER_COLD_MS)
        self.hover_timer.timeout.connect(self._hover_fire)
        self._prewarm_timer = QTimer()
        self._prewarm_timer.setSingleShot(True)
        self._prewarm_timer.setInterval(PREWARM_MS)
        self._prewarm_timer.timeout.connect(self._prewarm_fire)
        self.hover_pt = None
        self.markers = []
        self.busy = False
        self.queued = None            # latest request waiting while one is in flight or the image renders
        self.seq = 0                  # bumps on every click/move/reset; stale results are dropped
        self.notice = None            # last engine notice shown (e.g. it fell back to the CPU)
        self.window_px = WINDOW_PX
        self._next_window_px = None
        self.window = self.res = None
        self.image_id = self.image = self._render = None
        self.views = []               # (window, res, image_id, image) of finished windows, most recently used first
        self.drops = 0                # drop_views() calls: a window whose rendering spans one is not kept
        self._reset_object()
        canvas.scaleChanged.connect(self._zoomed)
        canvas.extentsChanged.connect(self._on_extents_changed)

    def _band(self, fill, stroke, width, style):
        b = QgsRubberBand(self.canvas, POLY)
        b.setFillColor(fill)
        b.setStrokeColor(stroke)
        b.setWidth(width)
        b.setLineStyle(style)
        return b

    # ------------------------------------------------------------ state
    def set_device(self, dev):
        self._device = dev

    def set_hover_enabled(self, on):
        self._hover_enabled = bool(on)
        if not self._hover_enabled:
            self.hover_timer.stop()
            self._clear_hover()

    def _schedule_prewarm(self):
        if self._alive and self._active and not self.points and not self.pending:
            self._prewarm_timer.start(PREWARM_MS)

    def _on_extents_changed(self):
        self._schedule_prewarm()

    def _cancel_prewarm(self, keep_encode_for=None):
        try:
            self._prewarm_timer.stop()
        except RuntimeError:
            pass
        if self._prewarm_render is not None:
            try:
                self._prewarm_render.cancelWithoutBlocking()
            except RuntimeError:
                pass
        self._prewarm_window = self._prewarm_res = self._prewarm_id = self._prewarm_render = None
        if self._prewarm_task is not None and self._prewarm_encoding_id != keep_encode_for:
            try:
                self._prewarm_task.cancel()
            except RuntimeError:
                pass
            self._prewarm_task = None
            self._prewarm_encoding_id = None

    def _prewarm_fire(self):
        if (not self._alive or not self._active or self.points or self.pending or self.busy
                or self._render is not None or self._prewarm_render is not None or self._prewarm_task is not None):
            return
        s = self.settings()
        layer = s["layer"]
        if layer is None:
            return
        ext = self.canvas.extent()
        if ext.isEmpty():
            return
        pt = ext.center()
        upp = self.canvas.mapUnitsPerPixel()
        if upp <= 0:
            return
        res = window_res(s["grid"], upp)
        if any(v[1] == res and self._inside(pt, INNER, v[0]) for v in self.views):
            return
        w, res = window_at(pt, s["grid"], upp, self.window_px)
        iid = uuid.uuid4().hex
        self._prewarm_window, self._prewarm_res, self._prewarm_id = w, res, iid
        self._prewarm_render = engine.render_window(
            layer, w, self.window_px, self._crs(),
            lambda img, n=self.drops, ww=w, rr=res, ii=iid: self._prewarm_rendered(ww, rr, ii, img, n),
        )

    def _prewarm_rendered(self, w, res, iid, img, drops):
        if not self._alive:
            return
        if w is self.window:                        # adopted by a user click while rendering
            self._rendered(w, img, drops)
            return
        if w is not self._prewarm_window:
            return
        self._prewarm_window = self._prewarm_res = self._prewarm_id = self._prewarm_render = None
        if drops != self.drops:
            return
        self.views = [(w, res, iid, img)] + self.views[:KEEP - 1]
        if (self._device in ("cuda", "mps") and self._active and not self.points
                and not self.pending and not self.busy and iid not in self._sent_ids):
            self._start_prewarm_encode(iid, img)

    def _start_prewarm_encode(self, image_id, image):
        self._prewarm_encoding_id = image_id
        task_ref = [None]

        def work(task):
            if task.isCanceled() or not self._alive:
                return None
            try:
                return engine._call(
                    "/encode",
                    {"image_id": image_id, "size": [image.width(), image.height()]},
                    timeout=30,
                    body=engine.rgba_bytes(image),
                )
            except Exception:
                return None

        def done(_exc, r=None):
            t = task_ref[0]
            if t is not None:
                QTimer.singleShot(0, lambda t=t: self._tasks.discard(t))
            if t is self._prewarm_task:
                self._prewarm_task = None
                self._prewarm_encoding_id = None
            if not self._alive or not r:
                return
            if r.get("device"):
                self._device = r["device"]
            if r.get("encoded"):
                self._mark_sent(image_id)

        task = QgsTask.fromFunction("Segmentot prewarm", work, on_finished=done, flags=_TASK_FLAGS)
        task_ref[0] = task
        self._prewarm_task = task
        self._tasks.add(task)
        QgsApplication.taskManager().addTask(task)

    def _clear_hover(self):
        self._hover_geom = None
        self.hover_band.reset(POLY)

    def _mark_sent(self, image_id):
        if not image_id:
            return
        self._sent_ids.add(image_id)
        if len(self._sent_ids) > KEEP * 2:
            keep = {v[2] for v in self.views}
            if self.image_id:
                keep.add(self.image_id)
            self._sent_ids &= keep
            self._sent_ids.add(image_id)

    def _reset_object(self):
        self.seq = getattr(self, "seq", 0) + 1
        self.queued = None
        self.points, self.labels, self.step_masks, self._step_history = [], [], [], []
        self.candidates, self.cand_idx = [], 0
        self.raw = self.geom = self.mask_id = None
        self.score = 0.0
        self.secs = 0.0
        self._recentered = False
        self._first_pt = None
        self._set_pending(False)
        self.preview.reset(POLY)
        self._clear_hover()
        scene = self.canvas.scene()
        for m in self.markers:
            if scene is not None:
                scene.removeItem(m)
        self.markers = []
        if self._next_window_px is not None:
            px, self._next_window_px = self._next_window_px, None
            if px != self.window_px:
                self.window_px = px
                self.views = []
                self._sent_ids.clear()
                self._forget_window()
        self._zoomed()                # the object may have kept a window of another zoom in use

    def set_window_px(self, px):
        px = int(px or WINDOW_PX)
        if px <= 0 or px == self.window_px:
            self._next_window_px = None
            return
        if self.points:
            self._next_window_px = px
            return
        self._next_window_px = None
        self.window_px = px
        self.new_imagery()

    def _set_pending(self, on):
        """pending: a click's outline for the current seq has not arrived yet (the cursor says so)."""
        self.pending = on
        self.setCursor(Qt.CursorShape.BusyCursor if on else Qt.CursorShape.CrossCursor)

    def _forget_window(self):
        if self._render is not None:
            self._render.cancelWithoutBlocking()
        self.window = self.image_id = self.image = self._render = None
        self.frame.reset(POLY)

    def _open_window(self, pt, force_new=False):
        s = self.settings()
        layer = s["layer"]
        if layer is None:
            self.status.emit("Choose an imagery layer in the Segmentot panel first.")
            return False
        self._forget_window()
        upp = self.canvas.mapUnitsPerPixel()
        res = window_res(s["grid"], upp)
        view = None if force_new else next(
            (v for v in self.views if v[1] == res and self._inside(pt, INNER, v[0])), None)
        if view is not None:
            self.views.remove(view)
            self.views.insert(0, view)
            self._cancel_prewarm(keep_encode_for=view[2])
            self.window, self.res, self.image_id, self.image = view
            self.frame.setToGeometry(QgsGeometry.fromRect(self.window), None)
            return True
        if (not force_new and self._prewarm_render is not None and self._prewarm_res == res
                and self._inside(pt, INNER, self._prewarm_window)):
            # Adopt the in-flight prewarm render directly so zero render work is wasted
            self.window, self.res, self.image_id, self.image = (
                self._prewarm_window, self._prewarm_res, self._prewarm_id, None)
            self._render = self._prewarm_render
            self._prewarm_window = self._prewarm_res = self._prewarm_id = self._prewarm_render = None
            self.frame.setToGeometry(QgsGeometry.fromRect(self.window), None)
            return True
        self._cancel_prewarm()
        self.window, self.res, self.image_id, self.image = \
            window_at(pt, s["grid"], upp, self.window_px) + (uuid.uuid4().hex, None)
        self.frame.setToGeometry(QgsGeometry.fromRect(self.window), None)
        w = self.window
        self._render = engine.render_window(layer, w, self.window_px, self._crs(),
                                            lambda img, n=self.drops: self._rendered(w, img, n))
        return True

    def _rendered(self, window, img, drops):
        if not self._alive or window is not self.window:     # unloaded, replaced or forgotten while rendering
            return
        if drops != self.drops and not self.points:
            self._forget_window()
            return
        self._render, self.image = None, img
        if drops == self.drops:
            self.views = [(window, self.res, self.image_id, img)] + self.views[:KEEP - 1]
        self._flush()

    def _zoomed(self, _scale=None):
        """Between objects, stop using a window that would show the AI a stale image: made for another zoom,
        or rendered before drop_views()."""
        if not self._alive:
            return
        if self.window is not None and not self.points and (
                window_res(self.settings()["grid"], self.canvas.mapUnitsPerPixel()) != self.res or
                self.image is not None and all(v[0] is not self.window for v in self.views)):
            self._forget_window()

    def _inside(self, pt, margin=0.0, w=None):
        w = self.window if w is None else w
        if w is None:
            return False
        return w.buffered(-margin * w.width()).contains(pt) if margin else w.contains(pt)

    def _to_px(self, pt):
        w = self.window
        return [(pt.x() - w.xMinimum()) / w.width() * self.window_px,
                (w.yMaximum() - pt.y()) / w.height() * self.window_px]

    def _hover_interval(self, pt):
        if self.image is not None and self._inside(pt, INNER):
            return HOVER_WARM_MS
        return HOVER_COLD_MS

    def _in_hover_geom(self, pt):
        g = self._hover_geom
        return g is not None and not g.isEmpty() and g.boundingBox().contains(pt) and g.contains(pt)

    # ------------------------------------------------------------ mouse / keys
    def canvasMoveEvent(self, e):
        if self.points or not self._hover_enabled:
            return
        pt = self.toMapCoordinates(e.pos())
        self.seq += 1                              # the preview for the old cursor position is stale
        self.hover_pt = pt
        if not self._in_hover_geom(pt):
            self._clear_hover()
        self.hover_timer.start(self._hover_interval(pt))

    def _hover_fire(self):
        if not self._alive or not self._hover_enabled:
            return
        pt = self.hover_pt
        if pt is None or self.points or self.pending:
            return
        if self.queued is not None and not self.queued[5]:
            return
        s = self.settings()
        if not s.get("hover", True):
            return
        if not self._inside(pt, INNER):
            if self.busy:                          # a new window would make the next click wait behind it
                self.hover_timer.start(HOVER_COLD_MS)
                return
            if s["layer"] is None or not self._open_window(pt):
                return
        self._request({"points": [self._to_px(pt)], "labels": [1]}, hover=True)

    def canvasReleaseEvent(self, e):
        self.hover_timer.stop()
        pt = self.toMapCoordinates(e.pos())
        positive = e.button() == Qt.MouseButton.LeftButton
        if e.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self._clear_hover()
            return
        # Instant 0 ms visual feedback: promote the cyan hover polygon to amber instead of flashing blank
        promoted = QgsGeometry(self._hover_geom) if (
            not self.points and positive and self._in_hover_geom(pt)) else None
        self._clear_hover()
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
            self._first_pt = pt
            self._recentered = False
            prev_mask = None
            if promoted is not None:
                self.preview.setToGeometry(promoted, None)
        else:
            prev_mask = self.mask_id
            if prev_mask is None and self.step_masks:
                prev_mask = self.step_masks[-1]
        self.points.append(self._to_px(pt))
        self.labels.append(1 if positive else 0)
        self.step_masks.append(prev_mask)
        self._step_history.append({"candidates": None, "cand_idx": 0, "secs": 0.0})
        m = QgsVertexMarker(self.canvas)
        m.setCenter(pt)
        m.setColor(QColor(40, 200, 90) if positive else QColor(230, 50, 50))
        m.setIconType(QgsVertexMarker.IconType.ICON_CIRCLE if positive else QgsVertexMarker.IconType.ICON_X)
        m.setIconSize(12)
        m.setPenWidth(3)
        self.markers.append(m)
        self._ask()

    def keyPressEvent(self, e):
        k = e.key()
        if k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.accept()
        elif k == Qt.Key.Key_Escape:
            self.clear()
        elif k in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            if self.points or not e.isAutoRepeat():    # holding the key must not delete saved objects
                self.undo()
        elif k == Qt.Key.Key_C and not (e.modifiers() & _MOD_MASK):
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
        self.step_masks.pop()
        if self._step_history:
            self._step_history.pop()
        self.mask_id = None
        m = self.markers.pop()
        scene = self.canvas.scene()
        if scene is not None:
            scene.removeItem(m)
        if self.points:
            prev = self._step_history[-1] if self._step_history else None
            if prev is not None and prev["candidates"]:
                self.seq += 1
                self.queued = None
                self._set_pending(False)
                self.candidates = list(prev["candidates"])
                self.cand_idx = min(prev["cand_idx"], len(self.candidates) - 1)
                self.secs = prev["secs"]
                self._use_candidate()
            else:
                self._ask()
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
        if self._step_history:
            self._step_history[-1]["cand_idx"] = self.cand_idx
        self._use_candidate()

    def refresh(self):
        """Re-apply refine options to the current object (called when the panel changes)."""
        if self.raw is not None:
            self._apply_refine()

    def resend(self):
        """Ask again for the current points (the Outline style changed)."""
        for step in self._step_history:
            step["candidates"] = None
            step["cand_idx"] = 0
        if self.points:
            self._ask()

    def activate(self):
        super().activate()
        self._active = True
        self._schedule_prewarm()

    def cleanup(self):
        """Cancel in-flight tasks/renders and disconnect signals before plugin unload."""
        self._alive = False
        self._active = False
        self.queued = None
        self.busy = False
        self._inflight_task = None
        self._inflight_hover = False
        self._inflight_image_id = None
        self._cancel_prewarm()
        try:
            self.hover_timer.stop()
        except RuntimeError:
            pass
        for t in list(self._tasks):
            try:
                t.cancel()
            except RuntimeError:
                pass
        try:
            self.canvas.scaleChanged.disconnect(self._zoomed)
        except (RuntimeError, TypeError):
            pass
        try:
            self.canvas.extentsChanged.disconnect(self._on_extents_changed)
        except (RuntimeError, TypeError):
            pass
        try:
            self._clear_hover()
            self.new_imagery()
            scene = self.canvas.scene()
            if scene is not None:
                for b in (self.preview, self.hover_band, self.frame):
                    scene.removeItem(b)
        except RuntimeError:
            pass

    def deactivate(self):
        self._active = False
        try:
            self._cancel_prewarm()
            self.hover_timer.stop()
            self._clear_hover()
            self.new_imagery()
        except RuntimeError:          # QGIS is closing and Qt already deleted the canvas items
            pass
        super().deactivate()

    def new_imagery(self):
        """Imagery or detail changed: no window matches any more."""
        self._cancel_prewarm()
        self.views = []
        self._sent_ids.clear()
        self._forget_window()
        self._reset_object()

    def drop_views(self):
        """The imagery looks different (its style changed): render windows again, except the one an object uses."""
        self._cancel_prewarm()
        self.views = []
        self.drops += 1
        if not self.points:
            self._sent_ids.clear()
            self._clear_hover()
            self._forget_window()
        else:
            self._sent_ids = {self.image_id} if self.image_id in self._sent_ids else set()

    # ------------------------------------------------------------ engine
    def _ask(self):
        """Request the outline of the current points."""
        self.seq += 1
        req = {"points": list(self.points), "labels": list(self.labels)}
        if len(self.points) == 1:
            req["progressive"] = True
        if self.step_masks and self.step_masks[-1] is not None:
            req["mask_id"] = self.step_masks[-1]
        step = self._step_history[-1] if self._step_history else None
        self._request(req, step=step)

    def _request(self, req, hover=False, step=None, quiet=False):
        if hover and self.queued is not None and not self.queued[5]:
            return                                 # real clicks always preempt queued hovers
        s = self.settings()
        payload = dict(req, image_id=self.image_id, outline=s["outline"], min_area_px=4)
        if hover:
            payload["hover"] = True
        else:
            self.hover_timer.stop()
            self._cancel_prewarm(keep_encode_for=self.image_id)
            self._set_pending(True)
            if not quiet:
                self.status.emit("Segmenting… (the first one loads the model, ~10 s)")
        self.queued = (payload, self.window, self.res, self.window_px, self.seq, hover, step)
        self._flush()

    def _flush(self):
        """Send the waiting request once nothing is in flight and its window's image is ready."""
        if not self._alive:
            return
        job = self.queued
        if job is None or self.image is None:
            return
        if self.busy:
            # Never make a real click wait behind a hover on another or already-encoded window
            is_click = not job[5]
            iid = job[0]["image_id"]
            if is_click and self._inflight_hover and (
                    self._inflight_image_id != iid or iid in self._sent_ids):
                if self._inflight_task is not None:
                    try:
                        self._inflight_task.cancel()
                    except RuntimeError:
                        pass
                self.busy = False
                self._inflight_task = None
                self._inflight_hover = False
                self._inflight_image_id = None
            else:
                return
        self.queued = None
        if job[1] is self.window and job[4] == self.seq:     # else its result would be dropped anyway
            self._send(job)

    def _send(self, job):
        payload, window, res, wpx, seq, hover, step = job
        image = self.image
        image_id = payload["image_id"]
        already_sent = image_id in self._sent_ids
        self.busy = True
        self._inflight_hover = bool(hover)
        self._inflight_image_id = image_id
        task_ref = [None]

        def work(task):
            if task.isCanceled() or not self._alive:
                return None
            if not already_sent:
                r = engine.call(
                    "/predict",
                    dict(payload, size=[image.width(), image.height()]),
                    body=engine.rgba_bytes(image),
                    cancelled=task.isCanceled,
                )
                return (r, False) if r is not None else None
            r = engine.call("/predict", payload, cancelled=task.isCanceled)
            if r is None:
                return None
            if r.get("need_image"):
                if task.isCanceled() or not self._alive:
                    return None
                r = engine.call(
                    "/predict",
                    dict(payload, size=[image.width(), image.height()]),
                    body=engine.rgba_bytes(image),
                    cancelled=task.isCanceled,
                )
                return (r, True) if r is not None else None
            return r, False

        def done(exc, result=None):
            t = task_ref[0]
            if t is not None:
                QTimer.singleShot(0, lambda t=t: self._tasks.discard(t))
            if not self._alive:
                return
            is_current = t is self._inflight_task
            if is_current:
                self.busy = False
                self._inflight_task = None
                self._inflight_hover = False
                self._inflight_image_id = None
            if t is not None and t.isCanceled():
                if not hover and seq == self.seq:
                    self._set_pending(False)
                self._flush()
                return
            if exc is not None:
                self._sent_ids.clear()
                if not hover and seq == self.seq:
                    self._set_pending(False)
                self._flush()
                self.status.emit(str(exc))
                return
            if result is None:
                if not hover and seq == self.seq:
                    self._set_pending(False)
                self._flush()
                return
            r, re_uploaded = result
            if r.get("device"):
                self._device = r["device"]
            if re_uploaded:
                self._sent_ids.clear()
            if r.get("stale"):
                self._flush()
                return
            if not r.get("need_image") and not r.get("error"):
                self._mark_sent(image_id)
            img_sz = r.get("image_size")
            if img_sz and not self.points:
                self.set_window_px(int(img_sz))
            if hover:
                self._flush()
                if not self.points and seq == self.seq:
                    self._show_hover(r, window, res, wpx)
            elif seq == self.seq:
                if r.get("needs_zoom") and self._stage1_zoom(r, window, res, wpx, step):
                    pass
                else:
                    self._set_pending(False)
                    self._flush()
                    self._show(r, window, res, wpx, step)
            else:
                self._flush()
            if img_sz and self.points:
                self.set_window_px(int(img_sz))
            if r.get("notice") and r["notice"] != self.notice:
                self.notice = r["notice"]
                self.status.emit(self.notice)

        task = QgsTask.fromFunction("Segmentot", work, on_finished=done, flags=_TASK_FLAGS)
        task_ref[0] = task
        self._task = task
        self._inflight_task = task
        self._tasks.add(task)
        QgsApplication.taskManager().addTask(task)

    def _stage1_zoom(self, result, window, res, wpx, step):
        """Show the fast coarse polygon immediately, then launch Stage-2 zoomed refinement."""
        cands = result.get("candidates") or []
        if not cands:
            return False
        raw = engine.pixel_polys_to_geometry(cands[0]["polygons"], window, wpx)
        if raw.isEmpty():
            return False
        if len(self.points) == 1 and not self._recentered and self._first_pt is not None:
            bb = raw.boundingBox()
            if not window.buffered(-res).contains(bb) and not self._inside(self._first_pt, 0.35, window):
                return False                        # let _show() auto-recenter instead of zooming clipped window
        g = engine.refine(raw, self.settings()["refine"], res, self._crs())
        if g is not None and not g.isEmpty():
            self.preview.setToGeometry(g, None)
        self.mask_id = cands[0].get("mask_id")
        self.status.emit("Refining outline…")
        req = {"points": list(self.points), "labels": list(self.labels), "refine_zoom": True}
        self._request(req, step=step, quiet=True)
        return True

    def _crs(self):
        return self.canvas.mapSettings().destinationCrs()

    def _show_hover(self, result, window, res, wpx=None):
        c = result.get("candidates") or []
        if not c:
            self._clear_hover()
            return
        wpx = self.window_px if wpx is None else wpx
        g = engine.refine(engine.pixel_polys_to_geometry(c[0]["polygons"], window, wpx),
                          self.settings()["refine"], res, self._crs())
        if g is not None and not g.isEmpty():
            self._hover_geom = g
            self.hover_band.setToGeometry(g, None)
        else:
            self._clear_hover()

    def _show(self, result, window, res, wpx=None, step=None):
        wpx = self.window_px if wpx is None else wpx
        self.candidates = [(engine.pixel_polys_to_geometry(c["polygons"], window, wpx), c["score"], c.get("mask_id"))
                           for c in result.get("candidates") or []]
        self.candidates = [c for c in self.candidates if not c[0].isEmpty()]
        self.cand_idx = 0
        self.secs = result.get("seconds", 0)
        if not self.candidates:
            self.raw = self.geom = self.mask_id = None
            self.preview.reset(POLY)
            self.status.emit(result.get("error") or "Nothing found there. Try another point.")
            return
        if len(self.points) == 1 and not self._recentered and self._first_pt is not None:
            bb = self.candidates[0][0].boundingBox()
            if not window.buffered(-res).contains(bb) and not self._inside(self._first_pt, 0.35, window):
                self._recentered = True
                if self._open_window(self._first_pt, force_new=True):
                    self.candidates = []
                    self.mask_id = None
                    self.points = [self._to_px(self._first_pt)]
                    self.step_masks = [None]
                    self._step_history = [{"candidates": None, "cand_idx": 0, "secs": 0.0}]
                    self._ask()
                    return
        if step is not None and self._step_history and self._step_history[-1] is step:
            step["candidates"] = list(self.candidates)
            step["cand_idx"] = self.cand_idx
            step["secs"] = self.secs
        self._use_candidate()

    def _use_candidate(self):
        self.raw, self.score, self.mask_id = self.candidates[self.cand_idx]
        self._apply_refine()

    def _apply_refine(self):
        s = self.settings()
        self.geom = engine.refine(self.raw, s["refine"], self.res, self._crs())
        self.preview.reset(POLY)
        if self.geom is None or self.geom.isEmpty():
            self.status.emit("Refine options removed the whole object (min area too big?).")
            return
        self.preview.setToGeometry(self.geom, None)
        alt = "  ·  outline {}/{} (C = next)".format(self.cand_idx + 1, len(self.candidates)) \
            if len(self.candidates) > 1 else ""
        bb = self.raw.boundingBox()
        cut = "" if self.window.buffered(-self.res).contains(bb) else \
            "\nThe object reaches the dotted frame and may be cut: zoom out and click it again."
        small = max(bb.width(), bb.height()) < 40 * self.res     # the model's mask cells are 4 window pixels wide
        if small and s["grid"] is not None and self.res == s["grid"][0]:    # zooming in stops at raster pixels
            cut += "\nSmall object: for a sharper outline set Detail to Map zoom (More options), " \
                "zoom in and click it again."
        elif small:
            cut += "\nSmall object: zoom in and click it again for a sharper outline."
        self.status.emit("Score {:.2f}  ·  {:.0f} m²{}  ({:.2f} s){}".format(
            self.score, engine.area_m2(self.geom, self._crs()), alt, self.secs, cut))
