"""Headless tests for engine.py and the window arithmetic of click_tool.py. The AI engine is not needed.

Run with QGIS's own Python, e.g. on macOS:
  QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 SEGMENTOT_TEST_PARENT=<dir holding the plugin as 'Segmentot'> \
      /Applications/QGIS.app/Contents/MacOS/python -s -P tests/test_engine.py
SEGMENTOT_TEST_PARENT defaults to the folder that contains this repo.
"""
import hmac
import json
import math
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

sys.path.insert(0, os.environ.get("SEGMENTOT_TEST_PARENT") or
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from osgeo import gdal  # noqa: E402
from qgis.core import (  # noqa: E402
    QgsApplication, QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsGeometry, QgsPointXY, QgsProject,
    QgsRasterLayer, QgsRasterRange, QgsRectangle,
)
from qgis.PyQt.QtCore import QEventLoop, QTimer  # noqa: E402

gdal.UseExceptions()
APP = QgsApplication([], False)
APP.initQgis()

from Segmentot.core import engine, runtime  # noqa: E402
from Segmentot.gui.click_tool import WINDOW_PX, window_at, window_res  # noqa: E402

CRS = QgsCoordinateReferenceSystem("EPSG:32633")      # metres
OPTS = {"expand": 0, "simplify": 0, "min_area_m2": 0, "fill_holes": False, "square": False, "smooth": False}


def ring(g):
    return g.asPolygon()[0][:-1]


def iou(a, b):
    inter = a.intersection(b)
    ia = inter.area() if inter is not None and not inter.isEmpty() else 0.0
    ua = a.area() + b.area() - ia
    return ia / ua if ua > 0 else 0.0


def count_right_angle_corners(pts, kx=1.0):
    corners = 0
    for i, p in enumerate(pts):
        u, v = pts[i - 1], pts[(i + 1) % len(pts)]
        ax, ay = (u.x() - p.x()) * kx, u.y() - p.y()
        bx, by = (v.x() - p.x()) * kx, v.y() - p.y()
        cos = (ax * bx + ay * by) / math.hypot(ax, ay) / math.hypot(bx, by)
        if abs(cos) < 0.02:
            corners += 1
        elif cos >= -0.999:
            return -1
    return corners


def make_rect(cx, cy, w, h, deg, kx=1.0):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    pts = [QgsPointXY(cx + (dx * c - dy * s) / kx, cy + dx * s + dy * c)
           for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]
    return QgsGeometry.fromPolygonXY([pts + [pts[0]]])


def rasterize(gt, origin_x=0.0, origin_y=0.0, px=1.0, pad=4):
    bb = gt.boundingBox()
    ix0 = int(math.floor((bb.xMinimum() - origin_x) / px)) - pad
    iy0 = int(math.floor((bb.yMinimum() - origin_y) / px)) - pad
    ix1 = int(math.ceil((bb.xMaximum() - origin_x) / px)) + pad
    iy1 = int(math.ceil((bb.yMaximum() - origin_y) / px)) + pad
    rows = []
    for iy in range(iy0, iy1):
        yc = origin_y + (iy + 0.5) * px
        xs = [ix for ix in range(ix0, ix1) if gt.contains(QgsPointXY(origin_x + (ix + 0.5) * px, yc))]
        if xs:
            run_s = prev = xs[0]
            for ix in xs[1:]:
                if ix != prev + 1:
                    rows.append(QgsGeometry.fromRect(QgsRectangle(
                        origin_x + run_s * px, origin_y + iy * px,
                        origin_x + (prev + 1) * px, origin_y + (iy + 1) * px)))
                    run_s = ix
                prev = ix
            rows.append(QgsGeometry.fromRect(QgsRectangle(
                origin_x + run_s * px, origin_y + iy * px,
                origin_x + (prev + 1) * px, origin_y + (iy + 1) * px)))
    return engine.polygons_only(QgsGeometry.unaryUnion(rows))


class PixelPolys(unittest.TestCase):
    def test_square_with_hole(self):
        window = QgsRectangle(1000, 2000, 1000 + WINDOW_PX * 0.5, 2000 + WINDOW_PX * 0.5)   # 0.5 m pixels
        top = window.yMaximum()
        sq = [[100, 100], [300, 100], [300, 300], [100, 300], [100, 100]]
        hole = [[150, 150], [200, 150], [200, 200], [150, 200], [150, 150]]
        g = engine.pixel_polys_to_geometry([[sq, hole]], window, WINDOW_PX)
        bb = g.boundingBox()
        self.assertEqual((bb.xMinimum(), bb.xMaximum()), (1050, 1150))
        self.assertEqual((bb.yMinimum(), bb.yMaximum()), (top - 150, top - 50))     # pixel rows go down
        self.assertEqual(g.area(), 100 * 100 - 25 * 25)
        self.assertEqual(len(g.asPolygon()), 2)
        self.assertFalse(g.contains(QgsPointXY(1000 + 175 * 0.5, top - 175 * 0.5)))  # inside the hole
        self.assertTrue(g.contains(QgsPointXY(1000 + 250 * 0.5, top - 250 * 0.5)))


class Refine(unittest.TestCase):
    def test_smooth_keeps_a_big_simple_shape(self):
        g = QgsGeometry.fromRect(QgsRectangle(0, 0, 800, 500))
        out = engine.refine(g, dict(OPTS, smooth=True), 1.0, CRS)
        self.assertGreaterEqual(out.area() / g.area(), 0.99)

    def test_square_corners_on_a_pixel_staircase(self):
        for deg in range(0, 50, 5):
            with self.subTest(deg=deg):
                gt = make_rect(100.0, 100.0, 100.0, 50.0, deg)
                stair = rasterize(gt)
                out = engine.refine(stair, dict(OPTS, square=True), 1.0, CRS)
                self.assertTrue(out.isGeosValid())
                self.assertEqual(count_right_angle_corners(ring(out)), 4)
                self.assertGreaterEqual(iou(out, gt), 0.98)
                self.assertAlmostEqual(out.area() / gt.area(), 1.0, delta=0.01)

    def test_square_corners_axis_aligned_with_rounded_corners(self):
        # 80x50 axis-aligned rectangle with 2-px chamfered corners like SAM outputs
        pts = [
            QgsPointXY(162, 175), QgsPointXY(162, 176), QgsPointXY(161, 176), QgsPointXY(161, 177),
            QgsPointXY(160, 177), QgsPointXY(160, 224), QgsPointXY(161, 224), QgsPointXY(161, 225),
            QgsPointXY(239, 225), QgsPointXY(239, 223), QgsPointXY(240, 223), QgsPointXY(240, 177),
            QgsPointXY(239, 177), QgsPointXY(239, 176), QgsPointXY(238, 176), QgsPointXY(238, 175),
            QgsPointXY(162, 175),
        ]
        g = QgsGeometry.fromPolygonXY([pts])
        gt = QgsGeometry.fromRect(QgsRectangle(160, 175, 240, 225))
        out = engine.refine(g, dict(OPTS, square=True), 1.0, CRS)
        self.assertEqual(count_right_angle_corners(ring(out)), 4)
        self.assertGreaterEqual(iou(out, gt), 0.99)
        self.assertAlmostEqual(out.area() / gt.area(), 1.0, delta=0.01)

    def test_square_corners_small_rotated_buildings(self):
        for w, h in ((12, 12), (16, 16), (20, 20), (30, 15)):
            with self.subTest(size=(w, h)):
                gt = make_rect(50.0, 50.0, float(w), float(h), 25)
                stair = rasterize(gt)
                out = engine.refine(stair, dict(OPTS, square=True), 1.0, CRS)
                self.assertEqual(count_right_angle_corners(ring(out)), 4)
                self.assertGreaterEqual(iou(out, gt), 0.95)
                self.assertAlmostEqual(out.area() / gt.area(), 1.0, delta=0.02)

    def test_square_corners_l_shape_and_courtyard(self):
        a = math.radians(25)
        c, s = math.cos(a), math.sin(a)
        l_local = [(0, 0), (80, 0), (80, 30), (35, 30), (35, 70), (0, 70), (0, 0)]
        l_gt = QgsGeometry.fromPolygonXY([[QgsPointXY(100 + x * c - y * s, 100 + x * s + y * c) for x, y in l_local]])
        l_out = engine.refine(rasterize(l_gt), dict(OPTS, square=True), 1.0, CRS)
        self.assertTrue(l_out.isGeosValid())
        self.assertEqual(count_right_angle_corners(ring(l_out)), 6)
        self.assertGreaterEqual(iou(l_out, l_gt), 0.96)

        a = math.radians(20)
        c, s = math.cos(a), math.sin(a)
        ext = [(-50, -40), (50, -40), (50, 40), (-50, 40), (-50, -40)]
        hole = [(-20, -15), (-20, 15), (20, 15), (20, -15), (-20, -15)]
        cy_gt = QgsGeometry.fromPolygonXY([
            [QgsPointXY(100 + x * c - y * s, 100 + x * s + y * c) for x, y in ext],
            [QgsPointXY(100 + x * c - y * s, 100 + x * s + y * c) for x, y in hole],
        ])
        cy_out = engine.refine(rasterize(cy_gt), dict(OPTS, square=True), 1.0, CRS)
        self.assertTrue(cy_out.isGeosValid())
        rings = cy_out.asPolygon()
        self.assertEqual(len(rings), 2)
        self.assertEqual(count_right_angle_corners(rings[0][:-1]), 4)
        self.assertEqual(count_right_angle_corners(rings[1][:-1]), 4)
        self.assertGreaterEqual(iou(cy_out, cy_gt), 0.96)

    def test_square_corners_geographic_crs_high_latitude(self):
        crs_geo = QgsCoordinateReferenceSystem("EPSG:4326")
        px_deg = 1e-5
        kx = math.cos(math.radians(60.0))
        for deg in (15, 30):
            with self.subTest(deg=deg):
                gt = make_rect(10.0, 60.0, 80 * px_deg, 50 * px_deg, deg, kx=kx)
                stair = rasterize(gt, origin_x=10.0, origin_y=60.0, px=px_deg)
                out = engine.refine(stair, dict(OPTS, square=True), px_deg, crs_geo)
                self.assertTrue(out.isGeosValid())
                self.assertEqual(count_right_angle_corners(ring(out), kx=kx), 4)
                self.assertGreaterEqual(iou(out, gt), 0.97)

    def test_square_corners_multipolygon_and_circle_safeguard(self):
        gt1 = make_rect(60.0, 60.0, 50.0, 30.0, 0)
        gt2 = make_rect(200.0, 60.0, 60.0, 30.0, 30)
        mp = QgsGeometry.unaryUnion([rasterize(gt1), rasterize(gt2)])
        out = engine.refine(mp, dict(OPTS, square=True), 1.0, CRS)
        self.assertTrue(out.isMultipart())
        parts = out.asMultiPolygon()
        self.assertEqual(len(parts), 2)
        for p in parts:
            self.assertEqual(count_right_angle_corners(p[0][:-1]), 4)

        circle = rasterize(QgsGeometry.fromPointXY(QgsPointXY(100, 100)).buffer(30.0, 32))
        sq_circle = engine.refine(circle, dict(OPTS, square=True), 1.0, CRS)
        self.assertGreaterEqual(iou(sq_circle, circle), 0.85)

    def test_expand(self):
        g = QgsGeometry.fromRect(QgsRectangle(0, 0, 100, 50))
        out = engine.refine(g, dict(OPTS, expand=2), 0.5, CRS)      # 2 px of 0.5 m = 1 m
        bb = out.boundingBox()
        self.assertEqual((bb.xMinimum(), bb.yMinimum(), bb.xMaximum(), bb.yMaximum()), (-1, -1, 101, 51))
        self.assertAlmostEqual(out.area(), 100 * 50 + 2 * (100 + 50) + 8 * math.sin(math.pi / 8), places=6)


class Render(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        path = os.path.join(cls.tmp.name, "img.tif")
        ds = gdal.GetDriverByName("GTiff").Create(path, 100, 100, 3, gdal.GDT_Byte)
        ds.SetGeoTransform((0, 1, 0, 100, 0, -1))           # x 0..100, y 0..100, 1 m pixels
        ds.SetProjection(CRS.toWkt())
        for b, v in ((1, 200), (2, 120), (3, 60)):
            band = ds.GetRasterBand(b)
            band.Fill(v)
            band.WriteRaster(60, 60, 20, 20, bytes(400))      # pure black block: x 60..80, y 20..40
        ds = None
        cls.layer = QgsRasterLayer(path, "img", "gdal")
        assert cls.layer.isValid()

    @classmethod
    def tearDownClass(cls):
        cls.layer = None
        cls.tmp.cleanup()

    def render(self, cancel=False, layer=None, extent=QgsRectangle(-50, -50, 150, 150), crs=CRS, size=WINDOW_PX):
        got, loop = [], QEventLoop()

        def on_done(img):
            got.append(img)
            loop.quit()
        job = engine.render_window(layer or self.layer, extent, size, crs, on_done)
        if cancel:
            job.cancelWithoutBlocking()
        QTimer.singleShot(20000, loop.quit)
        loop.exec()
        self.assertEqual(len(got), 1, "on_done was not called")
        return got[0]

    def check(self, img):
        self.assertEqual((img.width(), img.height()), (WINDOW_PX, WINDOW_PX))
        # window pixel = 200 / 1024 m: the raster covers pixels 256..768, the black block 563..665
        for x, y in ((100, 100), (900, 500), (500, 1000)):
            self.assertEqual(img.pixelColor(x, y).alpha(), 0, (x, y))
        inside = img.pixelColor(400, 400)
        self.assertEqual(inside.alpha(), 255)
        self.assertGreater(inside.red(), 0)
        black = img.pixelColor(614, 614)
        self.assertEqual((black.red(), black.green(), black.blue(), black.alpha()), (0, 0, 0, 255))

    def test_render_window(self):
        self.check(self.render())

    def test_rgba_bytes_in_a_thread(self):
        img, out = self.render(), []
        t = threading.Thread(target=lambda: out.append(engine.rgba_bytes(img)))
        t.start()
        t.join()
        raw = out[0]
        self.assertEqual(len(raw), WINDOW_PX * WINDOW_PX * 4)

        def px(x, y):
            i = (y * WINDOW_PX + x) * 4
            return tuple(raw[i:i + 4])
        self.assertEqual(px(100, 100)[3], 0)                                # no imagery
        self.assertEqual(px(400, 400), (200, 120, 60, 255))                 # row by row, R G B A
        self.assertEqual(px(614, 614), (0, 0, 0, 255))
        self.assertEqual(px(400, 614), (200, 120, 60, 255))                 # same column, beside the block

    def test_zoomed_out_raster_without_overviews_is_averaged(self):
        path = os.path.join(self.tmp.name, "checker.tif")
        ds = gdal.GetDriverByName("GTiff").Create(path, 400, 400, 3, gdal.GDT_Byte)
        ds.SetGeoTransform((0, 1, 0, 400, 0, -1))
        ds.SetProjection(CRS.toWkt())
        checker = bytes((x + y) % 2 * 255 for y in range(400) for x in range(400))     # 1-pixel checkerboard
        for b in (1, 2, 3):
            ds.GetRasterBand(b).WriteRaster(0, 0, 400, 400, checker)
        ds = None
        layer = QgsRasterLayer(path, "checker", "gdal")
        extent = QgsRectangle(0, 400 - 4 * WINDOW_PX, 4 * WINDOW_PX, 400)          # 4 raster pixels per window pixel
        img = self.render(layer=layer, extent=extent)
        reds = [img.pixelColor(x, y).red() for x in range(2, 98, 5) for y in range(2, 98, 5)]
        self.assertTrue(all(abs(r - 128) <= 8 for r in reds), sorted(set(reds)))       # grey, not black and white
        layer.dataProvider().setUserNoDataValue(1, [QgsRasterRange(0, 0)])             # "additional no data value"
        ds = gdal.Open(path, gdal.GA_Update)
        ds.GetRasterBand(1).WriteRaster(0, 0, 200, 400, bytes(200 * 400))              # left half: all no-data
        ds = None
        layer.dataProvider().reloadData()
        img = self.render(layer=layer, extent=extent)
        self.assertEqual((img.pixelColor(20, 50).alpha(), img.pixelColor(80, 50).alpha()), (0, 255))
        layer.dataProvider().enableProviderResampling(True)          # a resampling chosen by the user is kept
        self.assertIn(self.render(layer=layer, extent=extent).pixelColor(80, 50).green(), (0, 255))

    def test_zoomed_out_16x_and_reprojected_raster_is_averaged(self):
        width = 1024
        path = os.path.join(self.tmp.name, "checker16.tif")
        ds = gdal.GetDriverByName("GTiff").Create(path, width, width, 3, gdal.GDT_Byte)
        ds.SetGeoTransform((500000.0, 1.0, 0.0, 5500000.0 + width, 0.0, -1.0))
        ds.SetProjection(CRS.toWkt())
        checker = bytes((x + y) % 2 * 255 for y in range(width) for x in range(width))
        for b in (1, 2, 3):
            ds.GetRasterBand(b).WriteRaster(0, 0, width, width, checker)
        ds = None
        layer = QgsRasterLayer(path, "checker16", "gdal")
        self.assertTrue(layer.isValid())

        # 16x zoom-out without overviews (>8x default maxOversampling)
        size = 128
        ext16 = QgsRectangle(500000.0, 5500000.0 + width - 16 * size, 500000.0 + 16 * size, 5500000.0 + width)
        img16 = self.render(layer=layer, extent=ext16, crs=CRS, size=size)
        reds16 = [img16.pixelColor(x, y).red() for x in range(4, 60, 4) for y in range(4, 60, 4)]
        self.assertTrue(all(abs(r - 128) <= 8 for r in reds16), sorted(set(reds16)))

        # 4x zoomed-out reprojected UTM -> EPSG:3857
        crs_3857 = QgsCoordinateReferenceSystem("EPSG:3857")
        ct = QgsCoordinateTransform(CRS, crs_3857, QgsProject.instance().transformContext())
        center = ct.transformBoundingBox(layer.extent()).center()
        # At ~49.65 deg N, 4 m UTM ~ 6.18 m in EPSG:3857; 128 px * 6.18 m = 791 m
        half = 128 * 6.18 / 2.0
        ext_reproj = QgsRectangle(center.x() - half, center.y() - half, center.x() + half, center.y() + half)
        img_rp = self.render(layer=layer, extent=ext_reproj, crs=crs_3857, size=size)
        reds_rp = [img_rp.pixelColor(x, y).red() for x in range(20, 108, 6) for y in range(20, 108, 6)]
        self.assertTrue(all(abs(r - 128) <= 8 for r in reds_rp), sorted(set(reds_rp)))

    def test_native_grid_north_up_vs_rotated(self):
        self.assertEqual(engine.native_grid(self.layer, CRS), (1.0, 0.0, 100.0))
        self.assertIsNone(engine.native_grid(self.layer, QgsCoordinateReferenceSystem("EPSG:3857")))
        cases = (
            ("rot15", (500000.0, math.cos(math.radians(15)), -math.sin(math.radians(15)),
                       5500100.0, -math.sin(math.radians(15)), -math.cos(math.radians(15)))),
            ("rot90", (500000.0, 0.0, 1.0, 5500100.0, 1.0, 0.0)),
            ("southup", (500000.0, 1.0, 0.0, 5500000.0, 0.0, 1.0)),
        )
        for name, gt in cases:
            with self.subTest(name=name):
                path = os.path.join(self.tmp.name, "{}.tif".format(name))
                ds = gdal.GetDriverByName("GTiff").Create(path, 100, 100, 1, gdal.GDT_Byte)
                ds.SetGeoTransform(gt)
                ds.SetProjection(CRS.toWkt())
                ds = None
                lyr = QgsRasterLayer(path, name, "gdal")
                self.assertTrue(lyr.isValid())
                self.assertIsNone(engine.native_grid(lyr, CRS))

    def test_cancelled_render_still_calls_back(self):
        self.render(cancel=True)


class Trust(unittest.TestCase):
    """The plugin trusts a reply only when it carries the proof made with the key in the engine folder."""
    proof = staticmethod(lambda key, port, nonce: hmac.new(
        key.encode(), "{}:{}".format(port, nonce).encode(), "sha256").hexdigest())

    def setUp(self):
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        env = mock.patch.dict(os.environ, {"SEGMENTOT_HOME": home.name})
        env.start()
        self.addCleanup(env.stop)
        self.key = runtime.engine_key(create=True)
        test = self
        test.answer = lambda port, nonce: None        # the X-Segmentot-Proof the program on the port sends
        test.code = 200

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body = json.dumps({"ok": True, "engine": "segmentot", "error": "boom"}).encode()
                self.send_response(test.code)
                self.send_header("Content-Length", str(len(body)))
                proof = test.answer(self.server.server_port, self.headers.get("X-Segmentot-Nonce"))
                if proof:
                    self.send_header("X-Segmentot-Proof", proof)
                self.end_headers()
                self.wfile.write(body)
        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        old = dict(engine._cfg)
        self.addCleanup(engine._cfg.update, old)
        engine.configure(self.port, "base")

    def refused(self, call=None):
        with self.assertRaises(engine.EngineError) as e:
            (call or (lambda: engine._call("/health", timeout=5)))()
        self.assertNotIsInstance(e.exception, engine.EngineUnreachable)
        self.assertIn("another program", str(e.exception))

    def test_our_engine_is_trusted(self):
        self.answer = lambda port, nonce: self.proof(self.key, port, nonce)
        self.assertTrue(engine._call("/health", timeout=5)["ok"])
        self.assertTrue(engine.ensure_engine()["ok"])
        self.code = 500                               # its own error message is shown as it is
        with self.assertRaises(engine.EngineError) as e:
            engine._call("/health", timeout=5)
        self.assertEqual(str(e.exception), "boom")

    def test_another_program_on_the_port_is_refused(self):
        for name, answer in (("no proof", lambda port, nonce: None),
                             ("made up", lambda port, nonce: "0" * 64),
                             ("not ascii", lambda port, nonce: "\xe9" * 8),
                             ("another key", lambda port, nonce: self.proof("f" * 64, port, nonce)),
                             ("our engine on another port", lambda port, nonce: self.proof(self.key, port + 1, nonce)),
                             ("an old answer", lambda port, nonce: self.proof(self.key, port, "0" * 32))):
            with self.subTest(name):
                self.answer = answer
                self.refused()
                self.code = 500
                self.refused()
                self.code = 200
                with mock.patch.object(runtime, "start_engine") as start:
                    self.refused(engine.ensure_engine)        # and no second engine is started on that port
                self.assertFalse(start.called)

    def test_without_a_key_nothing_is_trusted(self):
        os.remove(os.path.join(runtime.home(), "engine.key"))
        self.answer = lambda port, nonce: self.proof("", port, nonce)
        self.refused()


class Window(unittest.TestCase):
    GRID = (0.03, 1000.0, 2000.0)                 # 3 cm raster pixels, origin (1000, 2000)
    PT = QgsPointXY(1012.3456, 1987.6543)

    def test_grid_window_follows_zoom_in_whole_raster_pixels(self):
        w, res = window_at(self.PT, self.GRID, 0.03 * 20.3)
        self.assertAlmostEqual(res, 0.03 * 20)
        self.assertAlmostEqual(w.width(), WINDOW_PX * 20 * 0.03)
        self.assertAlmostEqual(w.height(), w.width())
        for v in ((w.xMinimum() - 1000) / 0.03, (2000 - w.yMaximum()) / 0.03):
            self.assertAlmostEqual(v, round(v), places=6)                  # starts on a raster pixel edge
        self.assertLessEqual(abs(w.center().x() - self.PT.x()), 0.015 + 1e-9)
        self.assertLessEqual(abs(w.center().y() - self.PT.y()), 0.015 + 1e-9)

    def test_grid_window_never_finer_than_the_raster(self):
        w, res = window_at(self.PT, self.GRID, 0.001)
        self.assertEqual(res, 0.03)
        self.assertAlmostEqual(w.width(), WINDOW_PX * 0.03)

    def test_zoom_window(self):
        w, res = window_at(self.PT, None, 0.25)
        self.assertEqual(res, 0.25)
        for got, want in ((w.width(), 256), (w.height(), 256), (w.center().x(), self.PT.x()),
                          (w.center().y(), self.PT.y())):
            self.assertAlmostEqual(got, want, places=9)

    def test_window_res_changes_only_with_k(self):
        self.assertEqual(window_res(self.GRID, 0.03 * 2.2), window_res(self.GRID, 0.03 * 1.8))
        self.assertNotEqual(window_res(self.GRID, 0.03 * 2.2), window_res(self.GRID, 0.03 * 2.6))
        self.assertEqual(window_res(None, 0.7), 0.7)


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=2).result
    APP.exitQgis()
    sys.exit(not result.wasSuccessful())
