"""Talking to the local Segmentot engine, rendering map windows for it, and shaping results."""
import functools
import hmac
import http.client
import json
import math
import os
import secrets
import socket
import stat
import threading
import time
from collections import OrderedDict

from qgis.core import (
    Qgis,
    QgsCoordinateTransform,
    QgsCsException,
    QgsDistanceArea,
    QgsGeometry,
    QgsMapRendererParallelJob,
    QgsMapSettings,
    QgsPointXY,
    QgsProject,
    QgsRasterDataProvider,
    QgsRectangle,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QSize, Qt
from qgis.PyQt.QtGui import QColor, QImage

from . import runtime

_cfg = {"port": 8776, "model": "base"}
_cfg_lock = threading.Lock()
API = 2                 # request format of the engine this plugin talks to (server/segmentot_server.py has the same)
_start_lock = threading.Lock()
_local = threading.local()
_shutting_down = False

_STALE_CONN_ERRORS = (
    http.client.RemoteDisconnected,
    http.client.BadStatusLine,
    http.client.CannotSendRequest,
    http.client.ResponseNotReady,
    ConnectionResetError,
    ConnectionAbortedError,
    BrokenPipeError,
)


class EngineError(Exception):
    pass


class EngineUnreachable(EngineError):
    pass


def _get_cfg():
    with _cfg_lock:
        return _cfg["port"], _cfg["model"]


def _close_local_conn():
    conn = getattr(_local, "conn", None)
    _local.conn = None
    _local.conn_port = None
    if conn is not None:
        try:
            conn.close()
        except (OSError, http.client.HTTPException):
            pass


def _get_conn(port, timeout):
    """Per-thread persistent HTTP/1.1 connection directly to 127.0.0.1 (never through a system proxy)."""
    conn = getattr(_local, "conn", None)
    if conn is not None and (getattr(_local, "conn_port", None) != port or conn.sock is None):
        _close_local_conn()
        conn = None
    if conn is None:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        conn.connect()
        if conn.sock is not None:
            try:
                conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
        _local.conn = conn
        _local.conn_port = port
    else:
        conn.timeout = timeout
        if conn.sock is not None:
            conn.sock.settimeout(timeout)
    return conn


def set_shutting_down(val):
    global _shutting_down
    with _cfg_lock:
        _shutting_down = bool(val)
    if val:
        _close_local_conn()


def _cancelled(cancelled=None):
    with _cfg_lock:
        if _shutting_down:
            return True
    return bool(cancelled and cancelled())


def configure(port, model_key):
    port = int(port)
    with _cfg_lock:
        changed = _cfg["port"] != port
        _cfg["port"], _cfg["model"] = port, model_key
    if changed:
        _close_local_conn()


def url():
    port, _ = _get_cfg()
    return "http://127.0.0.1:{}".format(port)


def _check(headers, nonce, port=None):
    """Trust a reply only from this computer's engine: it alone knows the key in the engine folder, so a
    program that took the port cannot pass for it (and is never sent imagery). The port is part of the proof,
    so the reply of an engine on another port cannot be passed on either."""
    if port is None:
        port, _ = _get_cfg()
    key = runtime.engine_key()
    want = hmac.new((key or "").encode(), "{}:{}".format(port, nonce).encode(), "sha256").hexdigest()
    got = (headers.get("X-Segmentot-Proof") or "").encode("ascii", "replace")
    if not key or not hmac.compare_digest(want.encode(), got):
        raise EngineError("Port {} is used by another program (its reply is not from this computer's Segmentot "
                          "engine). Change the port in Segmentot's engine settings.".format(port))


def _call(path, payload=None, timeout=300, headers=None, body=None):
    """body: raw bytes to send; the payload then goes in the X-Segmentot-Request header."""
    port, _ = _get_cfg()
    extra = headers or {}
    if body is not None:
        data = body
        method = "POST"
        base_headers = {
            "Content-Type": "application/octet-stream",
            "X-Segmentot": "1",
            "X-Segmentot-Request": json.dumps(payload, separators=(",", ":")),
            "Connection": "keep-alive",
        }
    elif payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode()
        method = "POST"
        base_headers = {
            "Content-Type": "application/json",
            "X-Segmentot": "1",
            "Connection": "keep-alive",
        }
    else:
        data = None
        method = "GET"
        base_headers = {
            "Content-Type": "application/json",
            "X-Segmentot": "1",
            "Connection": "keep-alive",
        }
    for attempt in range(2):
        nonce = extra.get("X-Segmentot-Nonce") or secrets.token_hex(16)
        h = dict(base_headers)
        h["X-Segmentot-Nonce"] = nonce
        if extra:
            h.update(extra)
        resp = None
        try:
            conn = _get_conn(port, timeout)
            conn.request(method, path, body=data, headers=h)
            resp = conn.getresponse()
            status, reason, resp_headers = resp.status, resp.reason, resp.headers
            try:
                _check(resp_headers, nonce, port)
            except EngineError:
                resp.close()
                _close_local_conn()
                raise
            raw = resp.read()
            if resp.will_close:
                resp.close()
                _close_local_conn()
            break
        except _STALE_CONN_ERRORS as e:
            if resp is not None:
                resp.close()
            _close_local_conn()
            if attempt == 0 and not _cancelled():
                continue
            raise EngineUnreachable(str(e))
        except (http.client.HTTPException, OSError) as e:
            if resp is not None:
                resp.close()
            _close_local_conn()
            raise EngineUnreachable(str(e))
    if status >= 400:
        try:
            msg = json.loads(raw).get("error", "HTTP {} {}".format(status, reason))
        except Exception:
            msg = "HTTP {} {}".format(status, reason)
        raise EngineError(msg)
    return json.loads(raw)


def ensure_engine(wait_s=120, cancelled=None):
    """Make sure an engine answers on the port: use one already running, else start ours."""
    if _cancelled(cancelled):
        raise EngineError("Cancelled.")
    with _start_lock:
        if _cancelled(cancelled):
            raise EngineError("Cancelled.")
        try:
            return _call("/health", timeout=5)       # EngineError: another program has the port
        except EngineUnreachable:
            pass
        if _cancelled(cancelled):
            raise EngineError("Cancelled.")
        if not runtime.is_ready():
            raise EngineError("Segmentot's AI engine is not installed on this computer yet. "
                              "Click 'Install engine' in the Segmentot panel.")
        port, model = _get_cfg()
        if not runtime.model_downloaded(model):
            raise EngineError("The AI model is not downloaded yet. Click 'Download model' in the Segmentot panel.")
        if _cancelled(cancelled):
            raise EngineError("Cancelled.")
        runtime.start_engine(port, model)
        t0 = time.monotonic()
        while time.monotonic() - t0 < wait_s:
            if _cancelled(cancelled):
                with _cfg_lock:
                    stopping = _shutting_down
                if stopping:
                    runtime.stop_engine(wait=False)
                raise EngineError("Cancelled.")
            try:
                return _call("/health", timeout=5)
            except EngineUnreachable:
                if _cancelled(cancelled):
                    with _cfg_lock:
                        stopping = _shutting_down
                    if stopping:
                        runtime.stop_engine(wait=False)
                    raise EngineError("Cancelled.")
                if not runtime.engine_running_here():
                    raise EngineError("The engine stopped while starting. Last log lines:\n" +
                                      runtime.engine_log_tail())
                time.sleep(0.25)
        raise EngineError("The engine did not start within {} s. Log: {}".format(
            wait_s, os.path.join(runtime.home(), "engine.log")))


def call(path, payload=None, timeout=300, headers=None, body=None, cancelled=None):
    """Call the engine, starting it first if nothing answers."""
    if _cancelled(cancelled):
        raise EngineError("Cancelled.")
    try:
        return _call(path, payload, timeout, headers, body)
    except EngineUnreachable:
        if _cancelled(cancelled):
            raise EngineError("Cancelled.")
        ensure_engine(cancelled=cancelled)
        if _cancelled(cancelled):
            raise EngineError("Cancelled.")
        return _call(path, payload, timeout, headers, body)


def _zoom_out_ratio(layer, extent, size, crs):
    """Source raster pixels per output window pixel (0.0 if unknown or not applicable)."""
    if size <= 0 or extent.isEmpty():
        return 0.0
    rx, ry = layer.rasterUnitsPerPixelX(), layer.rasterUnitsPerPixelY()
    if rx <= 0 or ry <= 0:
        return 0.0
    px_x, px_y = extent.width() / size, extent.height() / size
    if layer.crs() == crs:
        return px_x / rx
    try:
        ct = QgsCoordinateTransform(crs, layer.crs(), QgsProject.instance().transformContext())
        sample = extent.center()
        try:
            overlap = extent.intersect(ct.transformBoundingBox(layer.extent(), Qgis.TransformDirection.Reverse))
        except QgsCsException:
            overlap = QgsRectangle()
        if not overlap.isEmpty():
            sample = overlap.center()
        p0 = ct.transform(sample)
        p1 = ct.transform(QgsPointXY(sample.x() + px_x, sample.y()))
        p2 = ct.transform(QgsPointXY(sample.x(), sample.y() + px_y))
    except QgsCsException:
        return 0.0
    d1 = math.hypot(p1.x() - p0.x(), p1.y() - p0.y())
    d2 = math.hypot(p2.x() - p0.x(), p2.y() - p0.y())
    if not (math.isfinite(d1) and math.isfinite(d2)):
        return 0.0
    return min(d1, d2) / max(rx, ry)


def render_window(layer, extent, size, crs, on_done):
    """Start rendering one layer over extent to a size x size image without blocking QGIS.

    on_done(QImage) runs later on the main thread; where there is no imagery the image is transparent.
    Returns the job: keep a reference while it runs; cancelWithoutBlocking() stops it (on_done still runs)."""
    p, rf, rend = layer.dataProvider(), layer.resampleFilter(), layer.renderer()
    if layer.providerType() == "gdal" and p is not None and rf is not None and rf.zoomedOutResampler() is None and \
            not p.isProviderResamplingEnabled() and rend is not None and \
            rend.type() in ("multibandcolor", "singlebandgray"):
        ratio = _zoom_out_ratio(layer, extent, size, crs)
        # 1.001: same-CRS rounding; 1.25: avoid blurring 1:1 reprojected views while averaging zoomed-out ones
        coarser = ratio > (1.001 if layer.crs() == crs else 1.25)
        if coarser:
            # zoomed out without overviews QGIS keeps single raster pixels: average them unless the user chose a
            # resampling (not class codes: no average class); scale maxOversampling so >8x zoom-out still averages
            layer = layer.clone()
            c = layer.dataProvider()
            for b in range(1, p.bandCount() + 1):      # clone() loses the no-data settings
                c.setUseSourceNoDataValue(b, p.useSourceNoDataValue(b))
                c.setUserNoDataValue(b, p.userNoDataValues(b))
            c.enableProviderResampling(True)
            avg = getattr(Qgis, "RasterResamplingMethod", QgsRasterDataProvider.ResamplingMethod).Average
            c.setZoomedOutResamplingMethod(avg)
            c.setMaxOversampling(max(8.0, min(64.0, float(math.ceil(ratio * 1.25)))))
            layer.setResamplingStage(Qgis.RasterResamplingStage.Provider)
    ms = QgsMapSettings()
    ms.setLayers([layer])
    ms.setDestinationCrs(crs)
    ms.setTransformContext(QgsProject.instance().transformContext())
    ms.setOutputSize(QSize(size, size))
    ms.setOutputDpi(96)
    ms.setExtent(extent)
    ms.setBackgroundColor(QColor(0, 0, 0, 0))
    job = QgsMapRendererParallelJob(ms)

    def done():
        nonlocal layer                 # the settings only hold a weak pointer: keep the clone until the end
        layer = None
        on_done(job.renderedImage())
        job.deleteLater()
    # queued: on_done never runs inside start() (it emits finished at once for invalid settings)
    job.finished.connect(done, Qt.ConnectionType.QueuedConnection)
    job.start()
    return job


def rgba_bytes(img):
    """QImage -> its pixels as RGBA8888 bytes, not premultiplied, row by row (alpha 0 = no imagery).
    Safe in a worker thread."""
    img = img.convertToFormat(QImage.Format.Format_RGBA8888)    # 32-bit rows are never padded
    n = img.width() * img.height() * 4
    if n <= 0:
        return b""
    bits = img.constBits()
    if hasattr(bits, "asstring"):
        return bits.asstring(n)
    if hasattr(bits, "asarray"):
        return memoryview(bits.asarray(n)).tobytes()
    return bytes(bits)[:n]


@functools.lru_cache(maxsize=64)
def _is_north_up_file(src, mtime_ns, size):
    try:
        from osgeo import gdal
        ds = gdal.Open(src, gdal.GA_ReadOnly)
        if ds is None:
            return True
        try:
            gt = ds.GetGeoTransform(can_return_null=True)
        finally:
            ds = None
    except Exception:
        return True
    if gt is None:
        return True
    tol_x = 1e-12 * max(abs(gt[1]), 1.0)
    tol_y = 1e-12 * max(abs(gt[5]), 1.0)
    return gt[1] > 0 and gt[5] < 0 and abs(gt[2]) <= tol_x and abs(gt[4]) <= tol_y


def _has_north_up_geotransform(src):
    """False if a local GDAL file has a rotated, sheared, or flipped GeoTransform."""
    if not src:
        return True
    try:
        st = os.stat(src)
    except OSError:
        return True
    if not stat.S_ISREG(st.st_mode):
        return True
    return _is_north_up_file(src, st.st_mtime_ns, st.st_size)


def native_grid(layer, crs):
    """(pixel size, origin x, origin y) of a local north-up raster with square pixels already in crs, else None."""
    if layer is None or layer.providerType() != "gdal" or layer.crs() != crs:
        return None
    rx, ry = layer.rasterUnitsPerPixelX(), layer.rasterUnitsPerPixelY()
    w, h = layer.width(), layer.height()
    if rx <= 0 or w <= 0 or h <= 0 or abs(rx - ry) > 1e-9 * max(rx, 1.0):
        return None
    e = layer.extent()
    if abs(e.width() - w * rx) > 1e-4 * max(e.width(), rx) or \
            abs(e.height() - h * ry) > 1e-4 * max(e.height(), ry):
        return None
    if not _has_north_up_geotransform(layer.source()):
        return None
    return rx, e.xMinimum(), e.yMaximum()


def polygons_only(geom):
    """Drop points/lines that makeValid() or buffer() can leave in a collection."""
    if geom is None or geom.isEmpty():
        return QgsGeometry()
    if geom.type() == QgsWkbTypes.GeometryType.PolygonGeometry and \
            QgsWkbTypes.flatType(geom.wkbType()) != QgsWkbTypes.Type.GeometryCollection:
        return geom
    polys = [g for g in geom.asGeometryCollection()
             if g.type() == QgsWkbTypes.GeometryType.PolygonGeometry and not g.isEmpty()]
    return QgsGeometry.unaryUnion(polys) if polys else QgsGeometry()


def pixel_polys_to_geometry(parts, window, size):
    """Pixel polygons ([[exterior, holes...], ...]) of an image covering window -> map geometry."""
    if not parts:
        return QgsGeometry()
    sx, sy = window.width() / size, window.height() / size
    x0, y0 = window.xMinimum(), window.yMaximum()
    polys = [[[QgsPointXY(x0 + x * sx, y0 - y * sy) for x, y in ring] for ring in rings]
             for rings in parts]
    geom = QgsGeometry.fromMultiPolygonXY(polys) if len(polys) > 1 else QgsGeometry.fromPolygonXY(polys[0])
    if not geom.isGeosValid():
        geom = geom.makeValid()
    return polygons_only(geom)


def distance_area(crs):
    """Per-thread LRU-cached QgsDistanceArea for crs and the current project ellipsoid."""
    proj = QgsProject.instance()
    ell = proj.ellipsoid()
    ellipsoid = ell if ell not in ("", "NONE", None) else "WGS84"
    key = (crs.toWkt() if crs is not None else "", ellipsoid)
    cache = getattr(_local, "da_cache", None)
    if cache is None:
        cache = OrderedDict()
        _local.da_cache = cache
    da = cache.get(key)
    if da is not None:
        cache.move_to_end(key)
        return da
    da = QgsDistanceArea()
    if crs is not None:
        da.setSourceCrs(crs, proj.transformContext())
    da.setEllipsoid(ellipsoid)
    cache[key] = da
    if len(cache) > 8:
        cache.popitem(last=False)
    return da


def area_m2(geom, crs):
    return distance_area(crs).measureArea(geom)


def _dedup_ring(pts, eps=1e-9):
    out = []
    for p in pts:
        if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > eps:
            out.append(p)
    if len(out) > 1 and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) <= eps:
        out.pop()
    return out


def _reanchor_ring(pts):
    """Start closed ring at its vertex farthest from the mean so DP never anchors on a mid-wall bump."""
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    k = max(range(len(pts)), key=lambda i: (pts[i][0] - mx) ** 2 + (pts[i][1] - my) ** 2)
    return pts[k:] + pts[:k]


def _dp_open(pts, start, end, tol2, keep):
    stack = [(start, end)]
    n = len(pts)
    while stack:
        a, b = stack.pop()
        if b - a <= 1:
            continue
        ax, ay = pts[a % n]
        bx, by = pts[b % n]
        dx, dy = bx - ax, by - ay
        l2 = dx * dx + dy * dy
        best_i, best_d2 = -1, -1.0
        for i in range(a + 1, b):
            x, y = pts[i % n]
            if l2 > 0.0:
                t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / l2))
                qx, qy = ax + t * dx, ay + t * dy
            else:
                qx, qy = ax, ay
            d2 = (x - qx) ** 2 + (y - qy) ** 2
            if d2 > best_d2:
                best_d2, best_i = d2, i
        if best_d2 > tol2:
            keep.add(best_i % n)
            stack.append((a, best_i))
            stack.append((best_i, b))


def _dp_ring_indices(pts, tol):
    n = len(pts)
    if n <= 3:
        return list(range(n))
    x0, y0 = pts[0]
    far = max(range(1, n), key=lambda i: (pts[i][0] - x0) ** 2 + (pts[i][1] - y0) ** 2)
    keep = {0, far}
    tol2 = tol * tol
    _dp_open(pts, 0, far, tol2, keep)
    _dp_open(pts, far, n, tol2, keep)
    return sorted(keep)


def _dominant_angle(ext, px):
    xs = [p[0] for p in ext]
    ys = [p[1] for p in ext]
    min_dim = min(max(xs) - min(xs), max(ys) - min(ys))
    tol = min(1.5 * px, max(0.75 * px, 0.12 * min_dim))
    idxs = _dp_ring_indices(ext, tol)
    if len(idxs) < 3:
        return 0.0
    segs = []
    m = len(idxs)
    for k in range(m):
        x0, y0 = ext[idxs[k]]
        x1, y1 = ext[idxs[(k + 1) % m]]
        length = math.hypot(x1 - x0, y1 - y0)
        if length > 1e-9:
            segs.append((length, math.atan2(y1 - y0, x1 - x0)))
    long_segs = [s for s in segs if s[0] >= 1.5 * px] or segs
    if not long_segs:
        return 0.0
    sx = sum(length * length * math.cos(4.0 * ang) for length, ang in long_segs)
    sy = sum(length * length * math.sin(4.0 * ang) for length, ang in long_segs)
    if math.hypot(sx, sy) < 1e-12:
        return 0.0
    return 0.25 * math.atan2(sy, sx)


def _extract_runs(pts, theta, px):
    """Rotate ring by -theta into (u, v), simplify staircases, group H/V runs, and drop micro-jogs."""
    c, s = math.cos(theta), math.sin(theta)
    uv = [(x * c + y * s, -x * s + y * c) for x, y in pts]
    us = [p[0] for p in uv]
    vs = [p[1] for p in uv]
    min_dim = min(max(us) - min(us), max(vs) - min(vs))
    if min_dim <= 1e-9:
        return None, 0.0
    tol = min(1.5 * px, max(0.5 * px, 0.12 * min_dim))
    idxs = _dp_ring_indices(uv, tol)
    if len(idxs) < 3:
        return None, min_dim
    n, m = len(uv), len(idxs)
    runs = []
    for k in range(m):
        i0, i1 = idxs[k], idxs[(k + 1) % m]
        orig_segs = []
        cur = i0
        while True:
            nxt = (cur + 1) % n
            orig_segs.append((uv[cur], uv[nxt]))
            cur = nxt
            if cur == i1:
                break
        du = uv[i1][0] - uv[i0][0]
        dv = uv[i1][1] - uv[i0][1]
        orient = "H" if abs(du) >= abs(dv) else "V"
        aligned = (abs(dv) <= 0.4 * abs(du)) if orient == "H" else (abs(du) <= 0.4 * abs(dv))
        edge = (orient, du, dv, aligned, orig_segs)
        if runs and runs[-1][0] == orient:
            runs[-1][1].append(edge)
        else:
            runs.append([orient, [edge]])
    if len(runs) > 1 and runs[0][0] == runs[-1][0]:
        runs[0][1] = runs[-1][1] + runs[0][1]
        runs.pop()
    if len(runs) < 4:
        return None, min_dim
    micro_tol = min(2.5 * px, 0.20 * min_dim)
    while len(runs) > 4:
        spans = [abs(sum(e[1] if r[0] == "H" else e[2] for e in r[1])) for r in runs]
        i = min(range(len(runs)), key=lambda j: spans[j])
        if spans[i] >= micro_tol:
            break
        nr = len(runs)
        if i == 0:
            runs = [[runs[nr - 1][0], runs[nr - 1][1] + runs[1][1]]] + runs[2:nr - 1]
        elif i == nr - 1:
            runs = [[runs[nr - 2][0], runs[nr - 2][1] + runs[0][1]]] + runs[1:nr - 2]
        else:
            runs[i - 1][1] = runs[i - 1][1] + runs[i + 1][1]
            del runs[i:i + 2]
    return runs, min_dim


def _chosen_segs(orient, edges):
    aligned = [e for e in edges if e[3]]
    idx_p = 0 if orient == "H" else 1
    total_span = sum(abs(e[1 + idx_p]) for e in edges)
    aligned_span = sum(abs(e[1 + idx_p]) for e in aligned)
    chosen = aligned if (aligned and aligned_span >= 0.5 * total_span) else edges
    return [seg for e in chosen for seg in e[4]]


def _refine_angle(ext_pts, theta0, px):
    """One-step pooled OLS residual angle refinement across the exterior walls (eliminates DP corner tilt)."""
    runs, _ = _extract_runs(ext_pts, theta0, px)
    if not runs:
        return theta0
    num, den = 0.0, 0.0
    for orient, edges in runs:
        segs = _chosen_segs(orient, edges)
        idx_p = 0 if orient == "H" else 1
        idx_c = 1 - idx_p
        sign = 1.0 if orient == "H" else -1.0
        ws = [abs(p1[idx_p] - p0[idx_p]) for p0, p1 in segs]
        wsum = sum(ws)
        if wsum < 3.0 * px:
            continue
        pbar = sum(w * 0.5 * (p0[idx_p] + p1[idx_p]) for (p0, p1), w in zip(segs, ws)) / wsum
        cbar = sum(w * 0.5 * (p0[idx_c] + p1[idx_c]) for (p0, p1), w in zip(segs, ws)) / wsum
        for (p0, p1), w in zip(segs, ws):
            if w <= 1e-12:
                continue
            dp = p1[idx_p] - p0[idx_p]
            dc = p1[idx_c] - p0[idx_c]
            pm = 0.5 * (p0[idx_p] + p1[idx_p]) - pbar
            cm = 0.5 * (p0[idx_c] + p1[idx_c]) - cbar
            num += sign * w * (pm * cm + dp * dc / 12.0)
            den += w * (pm * pm + dp * dp / 12.0)
    if den <= 1e-12:
        return theta0
    return theta0 + max(-0.2, min(0.2, math.atan2(num, den)))


def _run_coord(orient, edges):
    """Exact area-preserving constant wall coordinate via integral over the run's original segments."""
    idx_p = 0 if orient == "H" else 1
    idx_c = 1 - idx_p
    segs = _chosen_segs(orient, edges)
    num_s = sum(0.5 * (p0[idx_c] + p1[idx_c]) * (p1[idx_p] - p0[idx_p]) for p0, p1 in segs)
    den_s = sum(p1[idx_p] - p0[idx_p] for p0, p1 in segs)
    abs_den = sum(abs(p1[idx_p] - p0[idx_p]) for p0, p1 in segs)
    if abs(den_s) > 0.25 * abs_den and abs(den_s) > 1e-9:
        return num_s / den_s
    if abs_den > 1e-9:
        return sum(0.5 * (p0[idx_c] + p1[idx_c]) * abs(p1[idx_p] - p0[idx_p]) for p0, p1 in segs) / abs_den
    return sum(0.5 * (p0[idx_c] + p1[idx_c]) for p0, p1 in segs) / len(segs)


def _square_ring(ring_pts, theta, px):
    pts = _dedup_ring(ring_pts)
    if len(pts) < 3:
        return None, 0.0
    pts = _reanchor_ring(pts)
    runs, min_dim = _extract_runs(pts, theta, px)
    if not runs:
        return None, min_dim
    coords = [(r[0], _run_coord(r[0], r[1])) for r in runs]
    corners_uv = []
    for i in range(len(coords)):
        o0, c0 = coords[i - 1]
        _, c1 = coords[i]
        corners_uv.append((c1, c0) if o0 == "H" else (c0, c1))
    corners_uv = _dedup_ring(corners_uv, eps=1e-6 * px)
    if len(corners_uv) < 3:
        return None, min_dim
    clean = []
    nc = len(corners_uv)
    for i in range(nc):
        a, b, d = corners_uv[i - 1], corners_uv[i], corners_uv[(i + 1) % nc]
        if abs((b[0] - a[0]) * (d[1] - b[1]) - (b[1] - a[1]) * (d[0] - b[0])) > 1e-6 * px * px:
            clean.append(b)
    if len(clean) < 3:
        return None, min_dim
    c, s = math.cos(theta), math.sin(theta)
    back = [(u * c - v * s, u * s + v * c) for u, v in clean]
    back.append(back[0])
    return back, min_dim


def _geom_iou(a, b):
    aa, ba = a.area(), b.area()
    if aa <= 0 or ba <= 0:
        return 0.0
    inter = a.intersection(b)
    ia = inter.area() if inter is not None and not inter.isEmpty() else 0.0
    ua = aa + ba - ia
    return ia / ua if ua > 0 else 0.0


def _square_part(rings, px, kx):
    xs = [p.x() for p in rings[0]]
    ys = [p.y() for p in rings[0]]
    cx, cy = 0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys))
    conf_rings = [[((p.x() - cx) * kx, p.y() - cy) for p in r] for r in rings]
    orig_part = QgsGeometry.fromPolygonXY([[QgsPointXY(x, y) for x, y in r] for r in conf_rings])
    if not orig_part.isGeosValid():
        orig_part = polygons_only(orig_part.makeValid())
    if orig_part.isEmpty() or orig_part.area() <= 0:
        return QgsGeometry.fromPolygonXY(rings)

    ext_open = _dedup_ring(conf_rings[0])
    if len(ext_open) < 3:
        return QgsGeometry.fromPolygonXY(rings)
    ext_open = _reanchor_ring(ext_open)
    theta = _refine_angle(ext_open, _dominant_angle(ext_open, px), px)

    sq_ext, min_dim = _square_ring(conf_rings[0], theta, px)
    if sq_ext is None:
        return QgsGeometry.fromPolygonXY(rings)
    sq_rings = [sq_ext]
    for hole in conf_rings[1:]:
        sq_h, _ = _square_ring(hole, theta, px)
        if sq_h is not None:
            hg_orig = QgsGeometry.fromPolygonXY([[QgsPointXY(x, y) for x, y in hole]])
            hg_cand = QgsGeometry.fromPolygonXY([[QgsPointXY(x, y) for x, y in sq_h]])
            if hg_cand.isGeosValid() and _geom_iou(hg_cand, hg_orig) >= 0.75:
                sq_rings.append(sq_h)
                continue
        sq_rings.append(hole)

    cand = QgsGeometry.fromPolygonXY([[QgsPointXY(x, y) for x, y in r] for r in sq_rings])
    if not cand.isGeosValid():
        cand = polygons_only(cand.makeValid())
    if cand is None or cand.isEmpty() or cand.area() <= 0:
        return QgsGeometry.fromPolygonXY(rings)
    iou = _geom_iou(cand, orig_part)
    area_ratio = cand.area() / orig_part.area()
    min_iou = 0.80 if min_dim < 20.0 * px else 0.85
    if iou >= min_iou and 0.75 <= area_ratio <= 1.25:
        polys = cand.asMultiPolygon() if cand.isMultipart() else [cand.asPolygon()]
        out_polys = [[[QgsPointXY(cx + p.x() / kx, cy + p.y()) for p in r] for r in poly] for poly in polys]
        return QgsGeometry.fromMultiPolygonXY(out_polys) if len(out_polys) > 1 else QgsGeometry.fromPolygonXY(
            out_polys[0])
    return QgsGeometry.fromPolygonXY(rings)


def _square_polygon(geom, px, crs):
    """Regularize building footprints to their dominant orthogonal orientation, per part and with IoU safeguard."""
    if geom is None or geom.isEmpty() or px <= 0:
        return geom
    g = polygons_only(geom)
    if g.isEmpty():
        return geom
    is_geo = crs is not None and crs.isValid() and crs.isGeographic()
    polys = g.asMultiPolygon() if g.isMultipart() else [g.asPolygon()]
    out = []
    for rings in polys:
        if not rings or len(rings[0]) < 4:
            continue
        if is_geo:
            ys = [p.y() for p in rings[0]]
            kx = max(0.01, math.cos(math.radians(0.5 * (min(ys) + max(ys)))))
        else:
            kx = 1.0
        out.append(_square_part(rings, px, kx))
    if not out:
        return geom
    res = out[0] if len(out) == 1 else QgsGeometry.unaryUnion(out)
    return polygons_only(res) if res is not None and not res.isEmpty() else geom


def refine(geom, opts, px, crs):
    """Apply the panel's refine options. px = map units per image pixel (sizes are in pixels)."""
    if geom is None or geom.isEmpty():
        return geom
    g = QgsGeometry(geom)
    if opts.get("fill_holes"):
        g = g.removeInteriorRings()
    if opts.get("expand"):
        g = polygons_only(g.buffer(opts["expand"] * px, 4))
    if opts.get("simplify") and not g.isEmpty():
        s = g.simplify(opts["simplify"] * px)
        if s is not None and not s.isEmpty():
            if not s.isGeosValid():
                s = polygons_only(s.makeValid())
            if s is not None and not s.isEmpty():
                g = s
    if opts.get("square") and not g.isEmpty():
        g = _square_polygon(g, px, crs)
    if opts.get("smooth") and not g.isEmpty():
        # short segments first: smoothing cuts corners by a share of each edge's length
        g = g.densifyByDistance(2 * px).smooth(3, 0.25)
    min_area = opts.get("min_area_m2", 0)
    if min_area and not g.isEmpty():
        da = distance_area(crs)
        all_parts = g.asGeometryCollection()
        kept = [p for p in all_parts if da.measureArea(p) >= min_area]
        if not kept:
            g = QgsGeometry()
        elif len(kept) < len(all_parts):
            g = kept[0] if len(kept) == 1 else QgsGeometry.unaryUnion(kept)
    if not g.isEmpty() and not g.isGeosValid():
        g = polygons_only(g.makeValid())
    return g
