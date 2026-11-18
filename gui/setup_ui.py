"""Dialogs for installing and configuring Segmentot's engine."""
import os
import re
import traceback

from qgis.core import QgsApplication, QgsTask
from qgis.PyQt.QtCore import QSettings, Qt, QUrl, pyqtSignal
from qgis.PyQt.QtGui import QDesktopServices, QFont
from qgis.PyQt.QtWidgets import (
    QApplication, QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from ..core import runtime


PACK_FILTER = "Segmentot engine package (*.tar.gz)"


class GuideDialog(QDialog):
    """Ready-to-paste commands for an install that needs administrator rights."""

    def __init__(self, parent, intro=None):
        super().__init__(parent)
        self.setWindowTitle("Segmentot – permission help")
        self.setMinimumWidth(680)
        v = QVBoxLayout(self)
        head = QLabel(intro or "Segmentot installs into your own user folder and normally needs no administrator "
                               "rights. If the install stops with a permission error, run the matching command "
                               "below, then press <i>Install engine</i> again.")
        head.setWordWrap(True)
        v.addWidget(head)
        f = QFont("monospace")
        f.setStyleHint(QFont.StyleHint.TypeWriter)
        self.copy_buttons = []
        for title, how, cmd in runtime.permission_guide():
            lab = QLabel("<br><b>{}</b><br>{}".format(title, how))
            lab.setWordWrap(True)
            v.addWidget(lab)
            box = QPlainTextEdit(cmd)
            box.setReadOnly(True)
            box.setFont(f)
            box.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)     # one command per line, as it must be pasted
            box.setFixedHeight(box.fontMetrics().lineSpacing() * (cmd.count("\n") + 1) + 36)
            v.addWidget(box)
            b = QPushButton("Copy")
            b.clicked.connect(lambda _=False, c=cmd, b=b: (QApplication.clipboard().setText(c), b.setText("Copied ✓")))
            self.copy_buttons.append(b)
            row = QHBoxLayout()
            row.addStretch(1)
            row.addWidget(b)
            v.addLayout(row)
        foot = QLabel("Engine folder: <code>{}</code>".format(runtime.home()))
        foot.setWordWrap(True)
        v.addWidget(foot)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self.adjustSize()               # word-wrapped labels need the final width before the height is right


class TokenDialog(QDialog):
    """The steps for a model that Hugging Face only gives to accounts its maker approved (SAM 3)."""

    def __init__(self, parent, model_key):
        super().__init__(parent)
        mid, label, mb = runtime.MODELS[model_key]
        saved = os.path.exists(runtime.token_path())
        self.setWindowTitle("Segmentot – access to " + label.split(" (")[0])
        self.setMinimumWidth(520)
        v = QVBoxLayout(self)
        steps = (("<b>{}</b> is free, but its maker only releases it to Hugging Face accounts that asked for it. "
                  "Three steps, one time:".format(label.split(" (")[0]), None, None),
                 ("<b>1.</b> Log in on the model page (or make a free account) and ask for access. "
                  "The approval can take from minutes to days.", "Open the model page", runtime.model_docs(model_key)),
                 ("<b>2.</b> Make an access token of the type <i>Read</i>.", "Open the token page",
                  "https://huggingface.co/settings/tokens"),
                 ("<b>3.</b> Paste the token here:", None, None))
        for text, button, url in steps:
            lab = QLabel(text)
            lab.setWordWrap(True)
            v.addWidget(lab)
            if button:
                b = QPushButton(button)
                b.setToolTip(url)
                b.clicked.connect(lambda _=False, u=url: QDesktopServices.openUrl(QUrl(u)))
                v.addWidget(b, 0, Qt.AlignmentFlag.AlignLeft)
        self.token = QLineEdit()
        self.token.setEchoMode(QLineEdit.EchoMode.Password)
        self.token.setPlaceholderText("A token is already saved: leave empty to use it" if saved else "hf_…")
        v.addWidget(self.token)
        note = QLabel("<small>The token is kept in the engine folder on this computer and is only sent to "
                      "huggingface.co.</small>")
        note.setWordWrap(True)
        v.addWidget(note)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.go = bb.addButton("Download ({})".format(runtime.size_text(mb)), QDialogButtonBox.ButtonRole.AcceptRole)
        self.go.setEnabled(saved)
        self.token.textChanged.connect(lambda t: self.go.setEnabled(saved or bool(t.strip())))
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def accept(self):
        if self.token.text().strip():
            runtime.save_hf_token(self.token.text())
        super().accept()


TITLES = {"full": "install engine", "pack": "install engine from a package", "model": "download model",
          "export": "save engine package"}


class SetupTask(QgsTask):
    line = pyqtSignal(str)

    def __init__(self, mode, model_key, force_cpu, path, with_model):
        super().__init__("Segmentot: " + TITLES[mode], QgsTask.Flag.CanCancel)
        self.mode, self.model_key, self.force_cpu = mode, model_key, force_cpu
        self.path, self.with_model = path, with_model
        self.error = None
        self.cancelled = False

    def run(self):
        try:
            log = self.line.emit
            if self.mode == "export":
                runtime.export_pack(self.path, log, self.setProgress, self.isCanceled, self.with_model)
                return True
            if self.mode == "full":
                runtime.setup(log, self.setProgress, self.isCanceled, self.model_key, self.force_cpu)
            elif self.mode == "pack":
                runtime.install_pack(self.path, log, self.setProgress, self.isCanceled)
            if self.mode == "model" or (self.with_model and not runtime.model_downloaded(self.model_key)):
                self.setProgress(5 if self.mode == "model" else 99)
                runtime.download_model(self.model_key, log, self.isCanceled)
                st = runtime.load_state()
                st["model"] = self.model_key
                runtime.save_state(st)
            self.setProgress(100)
            return True
        except runtime.SetupError as e:
            self.error = str(e)
        except Exception:
            self.error = traceback.format_exc()
        self.cancelled = self.isCanceled()     # read here: the task may be deleted before the dialog hears of it
        return False


class SetupDialog(QDialog):
    finishedOk = pyqtSignal()
    failed = pyqtSignal()           # failed or cancelled

    def __init__(self, parent, mode, model_key, force_cpu=False, path=None, with_model=False):
        """mode: full (online install) · pack (install from the file path) · model (download model_key)
        · export (write the installed engine to path; with_model then means "with the models")."""
        super().__init__(parent)
        self.running = False
        self.mode = mode
        self.setWindowTitle("Segmentot – " + TITLES[mode])
        self.setMinimumWidth(560)
        v = QVBoxLayout(self)
        self.head = QLabel("Working… you can keep using QGIS." +
                           (" The PyTorch download is the slow part." if mode == "full" else ""))
        self.head.setWordWrap(True)
        v.addWidget(self.head)
        self.bar = QProgressBar()
        v.addWidget(self.bar)
        self.step = QLabel("")          # the newest log line: what it is doing right now
        v.addWidget(self.step)
        row = QHBoxLayout()
        self.details = QPushButton("Show details")
        self.details.setCheckable(True)
        self.details.toggled.connect(self._show_details)
        self.guide_btn = QPushButton("Permission help…")
        self.guide_btn.clicked.connect(lambda: GuideDialog(self).exec())
        self.guide_btn.setVisible(False)
        self.cancel = QPushButton("Cancel")
        self.cancel.clicked.connect(self._cancel)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        self.close_btn.setVisible(False)
        row.addWidget(self.details)
        row.addWidget(self.guide_btn)
        row.addStretch(1)
        row.addWidget(self.cancel)
        row.addWidget(self.close_btn)
        v.addLayout(row)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        f = QFont("monospace")
        f.setStyleHint(QFont.StyleHint.TypeWriter)
        self.log.setFont(f)
        self.log.setMinimumHeight(260)
        self.log.setVisible(False)
        v.addWidget(self.log, 1)
        self.task = SetupTask(mode, model_key, force_cpu, path, with_model)
        self.task.line.connect(self._line)
        self.task.progressChanged.connect(lambda p: self.bar.setValue(int(p)))
        self.task.taskCompleted.connect(self._ok)
        self.task.taskTerminated.connect(self._failed)

    def start(self):
        self.running = True
        self.show()
        QgsApplication.taskManager().addTask(self.task)

    def _line(self, text):
        self.log.appendPlainText(text)
        last = text.strip().splitlines()[-1:] or [""]
        self.step.setText(last[0][:80])

    def _show_details(self, on):
        self.details.setText("Hide details" if on else "Show details")
        self.log.setVisible(on)
        self.adjustSize()

    def _cancel(self):
        self.task.cancel()
        self.cancel.setEnabled(False)

    def _ended(self):
        self.running = False
        self.step.setText("")
        self.cancel.setVisible(False)
        self.close_btn.setVisible(True)
        self.show()                 # it may have been closed with the X while working
        self.raise_()

    def _ok(self):
        self.head.setText(
            "<b>Done.</b> Copy the file to a computer with the same system and choose <i>No internet? Install "
            "from a file…</i> in the Segmentot panel there." if self.mode == "export" else
            "<b>Done.</b> Go back to the Segmentot panel to continue.")
        self.bar.setValue(100)
        self._ended()
        self.finishedOk.emit()
        if self.mode != "export" and not self.details.isChecked():
            self.close()            # the panel shows the result; stay open only for someone reading the log

    def _failed(self):
        again = "try again from Settings." if self.mode == "export" else (
            "press the same button in the Segmentot panel again; downloads already done are reused.")
        if self.task.cancelled or self.task.error is None:      # error None: cancelled before it started
            self.log.appendPlainText("\nCancelled.")
            self.head.setText("<b>Cancelled.</b> To continue later, " + again)
        else:
            self.log.appendPlainText("\nFAILED: " + self.task.error)
            self.details.setChecked(True)
            if re.search("gated repo|is restricted|401 Client Error|403 Client Error", self.task.error):
                self.head.setText("<b>Hugging Face refused the download.</b> This model needs an approved access "
                                  "request and a valid token. Press its Download button again to see the steps; "
                                  "if you already asked for access, the approval may still be pending.")
            elif re.search("PermissionError|Permission denied|Access is denied|Operation not permitted|WinError 5",
                           self.task.error):
                self.guide_btn.setVisible(True)
                self.head.setText("<b>Setup needs a permission it does not have.</b> Press <i>Permission help…</i> "
                                  "for a ready-to-paste fix, then " + again)
            else:
                self.head.setText("<b>Setup did not finish.</b> The details below say why. Fix it (often the "
                                  "internet connection or disk space), then " + again)
        self._ended()
        self.failed.emit()


class EngineSettingsDialog(QDialog):
    """Two tabs. Models: every model with its size and whether it is on this computer, and one button to
    download or switch to it. Engine: where it is, what it runs on, port, reinstall, remove, package."""

    def __init__(self, parent, port, model_key):
        super().__init__(parent)
        self.setWindowTitle("Segmentot settings")
        self.setMinimumWidth(660)           # room for the longest model name next to its two buttons
        self.removed = False
        self.reinstall = False
        self.download = False           # the picked model has to be downloaded first
        self.deleted = False            # a model was deleted here: the panel has to look again
        self.export = None              # (path, with models) when the user asked for an engine package
        self.model_key = model_key
        installed = runtime.is_ready()
        tabs = QTabWidget()

        # --- Models
        page = QWidget()
        self.grid = QGridLayout(page)
        self.grid.setColumnStretch(0, 1)
        self.grid.setVerticalSpacing(10)
        self.rows = {}                  # model key -> (description, action button or "in use" label, delete button)
        self._fill_models()
        tabs.addTab(page, "Models")

        # --- Engine
        page = QWidget()
        f = QFormLayout(page)
        st = runtime.load_state()
        t = st.get("torch") or {}
        state = QLabel("Installed {} · PyTorch {}{}<br>Runs on: <b>{}</b>".format(
            st.get("finished", ""), t.get("torch", "–"),
            " (CUDA {})".format(t["cuda_build"]) if t.get("cuda_build") else "",
            t.get("gpu") or ("Apple GPU" if t.get("mps") else "CPU")) if installed else
            "Not installed yet: use step 1 in the Segmentot panel.")
        state.setWordWrap(True)
        f.addRow("Engine", state)
        folder = QLabel(runtime.home())
        folder.setWordWrap(True)
        folder.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        f.addRow("Folder", folder)
        self.port = QSpinBox()
        self.port.setRange(1024, 65535)
        self.port.setValue(port)
        self.port.setToolTip("Local port the engine listens on (127.0.0.1 only). Change it only if another "
                             "program uses this port.")
        f.addRow("Port", self.port)
        self.force_cpu = QCheckBox("Install the CPU version even if a GPU is found")
        self.force_cpu.setChecked(QSettings().value("Segmentot/force_cpu", False, type=bool))
        self.force_cpu.setToolTip("Smaller download. Takes effect on the next install or Reinstall….")
        self.force_cpu.setVisible(runtime.EDITION == "auto")       # a cpu/gpu edition decides by itself
        f.addRow(self.force_cpu)
        grid = QGridLayout()
        for i, (text, fn, on, tip) in enumerate((
                ("Open folder", lambda: self._open(runtime.home()), True, ""),
                ("Open engine log", lambda: self._open(os.path.join(runtime.home(), "engine.log")), True, ""),
                ("Reinstall…", self._reinstall, installed, "When the engine stopped working, or after changing "
                                                           "the graphics card"),
                ("Remove…", self._remove, True, "Delete the engine and its models from this computer"),
                ("Save engine package…", self._export, installed, "A file to install Segmentot on other computers "
                                                                  "without internet"),
                ("Permission help…", lambda: GuideDialog(self).exec(), True,
                 "Ready-to-paste commands for when the install needs administrator rights"))):
            b = QPushButton(text)
            b.setEnabled(on)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            grid.addWidget(b, i // 2, i % 2)
        f.addRow(grid)
        tabs.addTab(page, "Engine")

        v = QVBoxLayout(self)
        v.addWidget(tabs)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def _fill_models(self):
        while self.grid.count():
            self.grid.takeAt(0).widget().deleteLater()
        installed = runtime.is_ready()
        head = QLabel("Bigger models follow edges more exactly and are slower. One model is used at a time."
                      + ("" if installed else "<br>Models can be downloaded once the engine is installed "
                                              "(step 1 in the Segmentot panel)."))
        head.setWordWrap(True)
        self.grid.addWidget(head, 0, 0, 1, 3)
        for i, (k, (mid, label, mb)) in enumerate(runtime.MODELS.items(), 1):
            have = runtime.model_downloaded(k)
            text = QLabel("<b>{}</b><br><small>{} · {} · <a href='{}'>About this model</a></small>".format(
                label, runtime.size_text(mb),
                "<span style='color:#2e9e4f'>✓ downloaded</span>" if have else "not downloaded",
                runtime.model_docs(k)))
            text.setOpenExternalLinks(True)
            text.setToolTip(runtime.model_docs(k))
            if k == self.model_key and (have or not installed):
                act = QLabel("✓ In use" if have else "✓ Selected")
                act.setAlignment(Qt.AlignmentFlag.AlignCenter)
            else:
                act = QPushButton("Use" if have else "Download ({})".format(runtime.size_text(mb))
                                  if installed else "Select")
                act.clicked.connect(lambda _=False, k=k, dl=not have and installed: self._pick(k, dl))
            delete = QPushButton("Delete")
            delete.setToolTip("Remove this model from the computer to free {}".format(runtime.size_text(mb)))
            delete.setVisible(have)
            delete.clicked.connect(lambda _=False, k=k: self._delete(k))
            for col, w in enumerate((text, act, delete)):
                self.grid.addWidget(w, i, col)
            self.rows[k] = (text, act, delete)
        self.grid.setRowStretch(len(self.rows) + 1, 1)

    def _delete(self, key):
        label, mb = runtime.MODELS[key][1:]
        if QMessageBox.question(
                self, "Delete model",
                "Delete {} from this computer? This frees about {}. You can download it again later.{}".format(
                    label.split(" (")[0], runtime.size_text(mb),
                    "\n\nIt is the model in use: Segmentot will ask for a model again afterwards."
                    if key == self.model_key else "")) != QMessageBox.StandardButton.Yes:
            return
        if key == self.model_key:
            runtime.stop_engine()           # it holds the file open
        runtime.delete_model(key)
        self.deleted = True
        self._fill_models()

    def _pick(self, key, download):
        self.model_key, self.download = key, download
        self.accept()

    def values(self):
        return self.port.value(), self.model_key

    def _open(self, path):
        if os.path.exists(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Segmentot engine package",
                                              os.path.join(os.path.expanduser("~"), runtime.pack_name()), PACK_FILTER)
        if not path:
            return
        if not path.lower().endswith(".tar.gz"):
            path += ".tar.gz"
        models = QMessageBox.question(
            self, "Save Segmentot engine package",
            "Include the models already downloaded on this computer?\n\nYes: the other computers need "
            "no internet at all.\nNo: a smaller file; each computer downloads its model itself.") \
            == QMessageBox.StandardButton.Yes
        self.export = (path, models)
        self.accept()

    def _reinstall(self):
        if QMessageBox.question(self, "Reinstall Segmentot engine",
                                "Install the engine again? Use this when the engine stopped working or after "
                                "changing the graphics card. Downloads already done are reused.") \
                == QMessageBox.StandardButton.Yes:
            self.reinstall = True
            self.accept()

    def _remove(self):
        if QMessageBox.question(self, "Remove Segmentot engine",
                                "Delete {} (Python, PyTorch and models)? The plugin stays installed; "
                                "you can install the engine again later.".format(runtime.home())) \
                == QMessageBox.StandardButton.Yes:
            runtime.remove_all()
            self.removed = True
            self.accept()
