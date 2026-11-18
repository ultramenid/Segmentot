"""Headless tests for engine.py and the window arithmetic of click_tool.py. The AI engine is not needed.

Run with QGIS's own Python, e.g. on macOS:
  QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 SEGMENTOT_TEST_PARENT=<dir holding the plugin as 'Segmentot'> \
      /Applications/QGIS.app/Contents/MacOS/python -s -P tests/test_engine.py
SEGMENTOT_TEST_PARENT defaults to the folder that contains this repo.
"""
import math
import os
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.environ.get("SEGMENTOT_TEST_PARENT") or
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from osgeo import gdal  # noqa: E402
from qgis.core import (  # noqa: E402
    QgsApplication, QgsCoordinateReferenceSystem, QgsGeometry, QgsPointXY, QgsRasterLayer, QgsRectangle,
)
from qgis.PyQt.QtCore import QByteArray, QEventLoop, QTimer  # noqa: E402
from qgis.PyQt.QtGui import QImage  # noqa: E402

gdal.UseExceptions()
APP = QgsApplication([], False)
APP.initQgis()

from Segmentot.core import engine  # noqa: E402
from Segmentot.gui.click_tool import WINDOW_PX, window_at, window_res  # noqa: E402

CRS = QgsCoordinateReferenceSystem("EPSG:32633")      # metres
OPTS = {"expand": 0, "simplify": 0, "min_area_m2": 0, "fill_holes": False, "square": False, "smooth": False}


def ring(g):
    return g.asPolygon()[0][:-1]


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
        for deg in (20, 30):
            with self.subTest(deg=deg):
                stair = self.staircase(deg)
                self.assertGreater(len(ring(stair)), 50)
                out = engine.refine(stair, dict(OPTS, square=True), 1.0, CRS)
                pts = ring(out)
                corners = 0
                for i, p in enumerate(pts):
                    u, v = pts[i - 1], pts[(i + 1) % len(pts)]
                    ax, ay, bx, by = u.x() - p.x(), u.y() - p.y(), v.x() - p.x(), v.y() - p.y()
                    cos = (ax * bx + ay * by) / math.hypot(ax, ay) / math.hypot(bx, by)
                    if abs(cos) < 0.02:
                        corners += 1
                    else:
                        self.assertLess(cos, -0.999)           # any other vertex lies on a straight edge
                self.assertEqual(corners, 4)
                self.assertAlmostEqual(out.area() / stair.area(), 1, delta=0.03)

    @staticmethod
    def staircase(deg):
        """Pixel-edge outline of a 120 x 60 px rectangle rotated by deg, as the engine returns it."""
        a, cx, cy = math.radians(deg), 100.0, 100.0
        rows = []
        for y in range(200):
            xs = [x for x in range(200)
                  if abs((x + .5 - cx) * math.cos(a) + (y + .5 - cy) * math.sin(a)) <= 60 and
                  abs(-(x + .5 - cx) * math.sin(a) + (y + .5 - cy) * math.cos(a)) <= 30]
            if xs:
                rows.append(QgsGeometry.fromRect(QgsRectangle(xs[0], y, xs[-1] + 1, y + 1)))
        r = ring(QgsGeometry.unaryUnion(rows))
        i = min(range(len(r)), key=lambda k: (r[k].y(), r[k].x()))   # GDAL/rasterio start at the first pixel
        return QgsGeometry.fromPolygonXY([r[i:] + r[:i] + [r[i]]])

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

    def render(self, cancel=False):
        got, loop = [], QEventLoop()

        def on_done(img):
            got.append(img)
            loop.quit()
        job = engine.render_window(self.layer, QgsRectangle(-50, -50, 150, 150), WINDOW_PX, CRS, on_done)
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

    def test_encode_png_in_a_thread_keeps_alpha(self):
        img, out = self.render(), []
        t = threading.Thread(target=lambda: out.append(engine.encode_png(img)))
        t.start()
        t.join()
        back = QImage.fromData(QByteArray.fromBase64(out[0].encode()), "PNG")
        self.assertTrue(back.hasAlphaChannel())
        self.check(back)

    def test_cancelled_render_still_calls_back(self):
        self.render(cancel=True)


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
