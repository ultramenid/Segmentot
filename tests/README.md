# Tests

Three unit-test files, one end-to-end test, two smaller real-engine checks and a Qt6 check. All use stdlib `unittest`; none needs pytest.

| File | Needs | Run from the plugin folder |
|---|---|---|
| `test_runtime.py` | any Python 3 | `python3 tests/test_runtime.py` |
| `test_server.py` | the installed engine | `~/.segmentot/venv/bin/python -m unittest tests/test_server.py` |
| `test_engine.py` | QGIS's Python | `$QGIS_PY tests/test_engine.py` |
| `e2e.py` | QGIS's Python + the installed engine (it installs it if missing) | `$QGIS_PY tests/e2e.py` |
| `ui_smoke.py` | QGIS's Python + an engine package file (Settings → Save engine package…) | `$QGIS_PY tests/ui_smoke.py <package.tar.gz>` |
| `fallback_real.py` | the installed engine | `~/.segmentot/venv/bin/python tests/fallback_real.py` |
| `qt6_check.py` | any Python with PyQt6 (`pip install PyQt6`) | `python tests/qt6_check.py` |

`e2e.py` drives the real panel and map tool headless against the real engine and model, on a synthetic
image with objects at known positions.
`ui_smoke.py` drives the panel's two setup steps, the startup check, the permission guide and the progress
window, and installs the package into a temporary engine folder (its engine uses its own port).
`fallback_real.py` loads the real model on the GPU, injects a GPU failure and checks the CPU takes over.
`SEGMENTOT_TEST_MODEL=edge $QGIS_PY tests/e2e.py` runs the end-to-end test with another model
(the accuracy limits are set for Base+: EdgeTAM misses "Square corners" by one corner point).

`$QGIS_PY` on macOS (QGIS.app bundle):

```sh
R=/Applications/QGIS.app/Contents
QGIS_PY="env PYTHONHOME=$R/Resources \
  PYTHONPATH=$R/Resources/python3.11:$R/Resources/python3.11/lib-dynload:$R/Resources/python3.11/site-packages \
  QGIS_PREFIX_PATH=$R/MacOS PROJ_DATA=$R/Resources/qgis/proj GDAL_DATA=$R/Resources/qgis/gdal \
  QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 $R/MacOS/python -s -P"
```

On Linux and Windows use the Python that ships with QGIS and set `QT_QPA_PLATFORM=offscreen`.
The plugin is imported as a package from its parent folder; set `SEGMENTOT_TEST_PARENT` to use another one.

`qt6_check.py` scans the plugin for Qt names that PyQt6 does not have, then builds the panel and every dialog
on real Qt 6 widgets with stand-ins for the QGIS classes. It does not replace a run in a QGIS built with Qt6.

Before a release, from the plugin folder: `flake8 .` and `bandit -r . -x ./tests` must both be clean,
then `scripts/make_zip.sh` builds the zip.
