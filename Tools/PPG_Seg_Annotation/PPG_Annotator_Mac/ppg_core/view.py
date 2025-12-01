from PyQt6.QtWidgets import (QMainWindow, QVBoxLayout, QWidget, 
                             QToolBar, QStatusBar, QLabel, QMessageBox,
                             QDialog, QFormLayout, QLineEdit, QDialogButtonBox)
from PyQt6.QtGui import QAction, QCursor, QActionGroup, QIntValidator
from PyQt6.QtCore import Qt
import pyqtgraph as pg

class SegmentationDialog(QDialog):
    def __init__(self, parent=None, default_win=256, default_stride=128):
        super().__init__(parent)
        self.setWindowTitle("Data Settings")
        self.resize(300, 150)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        
        self.inp_win = QLineEdit(str(default_win))
        self.inp_win.setValidator(QIntValidator(1, 100000))
        self.inp_stride = QLineEdit(str(default_stride))
        self.inp_stride.setValidator(QIntValidator(1, 100000))
        
        form.addRow("Window Size:", self.inp_win)
        form.addRow("Stride:", self.inp_stride)
        layout.addLayout(form)
        
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_values(self):
        return int(self.inp_win.text()), int(self.inp_stride.text())

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PPG Annotator Pro")
        self.resize(1200, 800)
        
        central = QWidget()
        self.setCentralWidget(central)
        self.layout = QVBoxLayout(central)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1e1e1e')
        self.plot_widget.getPlotItem().setDownsampling(mode='peak', auto=True)
        self.plot_widget.getPlotItem().setClipToView(True)
        self.layout.addWidget(self.plot_widget)
        
        self._setup_toolbar()
        self._setup_statusbar()

    def _setup_toolbar(self):
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        self.action_load = QAction("📂 Load", self)
        self.action_theme = QAction("🌗 Theme", self)
        
        self.action_nav = QAction("Navigation", self)
        self.action_nav.setCheckable(True)
        self.action_nav.setChecked(True)
        self.action_paint = QAction("Anottate", self)
        self.action_paint.setCheckable(True)
        self.action_erase = QAction("Eraser", self)
        self.action_erase.setCheckable(True)

        self.mode_group = QActionGroup(self)
        self.mode_group.addAction(self.action_nav)
        self.mode_group.addAction(self.action_paint)
        self.mode_group.addAction(self.action_erase)
        self.mode_group.setExclusive(True)
        
        self.action_auto = QAction("Auto-Detect", self)
        self.action_clear_auto = QAction("Reset", self)
        self.action_save = QAction("💾 Export Mask", self)
        
        toolbar.addAction(self.action_load)
        toolbar.addAction(self.action_theme)
        toolbar.addSeparator()
        toolbar.addAction(self.action_nav)
        toolbar.addAction(self.action_paint)
        toolbar.addAction(self.action_erase)
        toolbar.addSeparator()
        toolbar.addAction(self.action_auto)
        toolbar.addAction(self.action_clear_auto)
        toolbar.addSeparator()
        toolbar.addAction(self.action_save)

    def _setup_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.lbl_info = QLabel("Ready")
        self.status_bar.addWidget(self.lbl_info)

    def apply_plot_theme(self, dark=True):
        self.plot_widget.setBackground('#1e1e1e' if dark else 'w')

    def show_info(self, msg):
        QMessageBox.information(self, "Info", msg)
        
    def set_cursor(self, cursor_type):
        self.plot_widget.setCursor(QCursor(cursor_type))
