"""Talking to the local Segmentot engine, rendering map windows for it, and shaping results."""
import base64
import hmac
import json
import os
import secrets
import threading
import time
import urllib.error
import urllib.request

from qgis.core import (
    QgsDistanceArea,
    QgsGeometry,
    QgsMapRendererParallelJob,
    QgsMapSettings,
    QgsPointXY,
    QgsProject,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QBuffer, QByteArray, QIODevice, QSize, Qt
from qgis.PyQt.QtGui import QColor

from . import runtime

# localhost must never go through a system proxy
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_cfg = {"port": 8776, "model": "base"}
_start_lock = threading.Lock()


class EngineError(Exception):
    pass


class EngineUnreachable(EngineError):
    pass


def configure(port, model_key):
    _cfg["port"], _cfg["model"] = int(port), model_key


def url():
    return "http://127.0.0.1:{}".format(_cfg["port"])


def _check(headers, nonce):
    """Trust a reply only from this computer's engine: it alone knows the key in the engine folder, so a
    program that took the port cannot pass for it (and is never sent imagery). The port is part of the proof,
    so the reply of an engine on another port cannot be passed on either."""
    key = runtime.engine_key()
    want = hmac.new((key or "").encode(), "{}:{}".format(_cfg["port"], nonce).encode(), "sha256").hexdigest()
    got = (headers.get("X-Segmentot-Proof") or "").encode("ascii", "replace")
    if not key or not hmac.compare_digest(want.encode(), got):
        raise EngineError("Port {} is used by another program (its reply is not from this computer's Segmentot "
                          "engine). Change the port in Segmentot's engine settings.".format(_cfg["port"]))


def _call(path, payload=None, timeout=300, headers=None):
    data = None if payload is None else json.dumps(payload).encode()
    nonce = secrets.token_hex(16)
    h = {"Content-Type": "application/json", "X-Segmentot": "1", "X-Segmentot-Nonce": nonce}
    h.update(headers or {})
    req = urllib.request.Request(url() + path, data=data, headers=h)
    try:
        with _opener.open(req, timeout=timeout) as r:
            _check(r.headers, nonce)
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        _check(e.headers, nonce)
        try:
            msg = json.loads(e.read()).get("error", str(e))
        except Exception:
            msg = str(e)
        raise EngineError(msg)
    except (urllib.error.URLError, ConnectionError, OSError) as e:
        raise EngineUnreachable(str(e))


def ensure_engine(wait_s=120):
    """Make sure an engine answers on the port: use one already running, else start ours."""
    with _start_lock:
        try:
            return _call("/health", timeout=5)       # EngineError: another program has the port
        except EngineUnreachable:
            pass
        if not runtime.is_ready():
            raise EngineError("Segmentot's AI engine is not installed on this computer yet. "
                              "Click 'Install engine' in the Segmentot panel.")
        if not runtime.model_downloaded(_cfg["model"]):
            raise EngineError("The AI model is not downloaded yet. Click 'Download model' in the Segmentot panel.")
        runtime.start_engine(_cfg["port"], _cfg["model"])
        t0 = time.time()
        while time.time() - t0 < wait_s:
            try:
                return _call("/health", timeout=5)
            except EngineUnreachable:
                if not runtime.engine_running_here():
                    raise EngineError("The engine stopped while starting. Last log lines:\n" +
                                      runtime.engine_log_tail())
                time.sleep(0.5)
        raise EngineError("The engine did not start within {} s. Log: {}".format(
            wait_s, os.path.join(runtime.home(), "engine.log")))


def call(path, payload=None, timeout=300, headers=None):
    """Call the engine, starting it first if nothing answers."""
    try:
        return _call(path, payload, timeout, headers)
    except EngineUnreachable:
        ensure_engine()
        return _call(path, payload, timeout, headers)


def render_window(layer, extent, size, crs, on_done):
    """Start rendering one layer over extent to a size x size image without blocking QGIS.

    on_done(QImage) runs later on the main thread; where there is no imagery the image is transparent.
    Returns the job: keep a reference while it runs; cancelWithoutBlocking() stops it (on_done still runs)."""
    ms = QgsMapSettings()
    ms.setLayers([layer])
    ms.setDestinationCrs(crs)
    ms.setTransformContext(QgsProject.instance().transformContext())
    ms.setExtent(extent)
    ms.setOutputSize(QSize(size, size))
    ms.setOutputDpi(96)
    ms.setBackgroundColor(QColor(0, 0, 0, 0))
    job = QgsMapRendererParallelJob(ms)

    def done():
        on_done(job.renderedImage())
        job.deleteLater()
    # queued: on_done never runs inside start() (it emits finished at once for invalid settings)
    job.finished.connect(done, Qt.ConnectionType.QueuedConnection)
    job.start()
    return job


def encode_png(img):
    """QImage -> base64 PNG with its alpha channel (transparent = no imagery). Safe in a worker thread."""
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG", 100)          # 100 = no compression: 5x faster, and it only goes to localhost
    return base64.b64encode(bytes(ba)).decode()


def native_grid(layer, crs):
    """(pixel size, origin x, origin y) of a local raster with square pixels already in crs, else None."""
    if layer is None or layer.providerType() != "gdal" or layer.crs() != crs:
        return None
    rx, ry = layer.rasterUnitsPerPixelX(), layer.rasterUnitsPerPixelY()
    if rx <= 0 or abs(rx - ry) > 1e-9 * max(rx, 1.0):
        return None
    e = layer.extent()
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


def area_m2(geom, crs):
    da = QgsDistanceArea()
    da.setSourceCrs(crs, QgsProject.instance().transformContext())
    da.setEllipsoid(QgsProject.instance().ellipsoid() if QgsProject.instance().ellipsoid() not in ("", "NONE")
                    else "WGS84")
    return da.measureArea(geom)


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
            g = s
    if opts.get("square") and not g.isEmpty():
        # pixel-edge steps are already right angles and the model rounds corners by 2-3 px:
        # drop both first (2.5 px measured on real outlines; 1.5 px left chamfered corners)
        o = g.simplify(2.5 * px).orthogonalize(1.0e-8, 1000, 20.0)
        if o is not None and not o.isEmpty() and o.isGeosValid():
            g = o
    if opts.get("smooth") and not g.isEmpty():
        # short segments first: smoothing cuts corners by a share of each edge's length
        g = g.densifyByDistance(2 * px).smooth(3, 0.25)
    min_area = opts.get("min_area_m2", 0)
    if min_area and not g.isEmpty():
        parts = [p for p in g.asGeometryCollection() if area_m2(p, crs) >= min_area]
        g = QgsGeometry.unaryUnion(parts) if parts else QgsGeometry()
    if not g.isEmpty() and not g.isGeosValid():
        g = polygons_only(g.makeValid())
    return g
