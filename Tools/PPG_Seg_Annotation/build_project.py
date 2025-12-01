import os

project_name = "PPG_Annotator_Mac"

# Content for the files
structure = {
    "requirements.txt": """numpy>=1.23.0
PyQt6>=6.4.0
pyqtgraph>=0.13.1
pandas>=1.5.0
scipy>=1.9.0""",

    # MAIN.PY - NOW INCLUDES THE MACOS FIX
    "main.py": """import sys
import os
import logging

def macos_plugin_fix():
    \"\"\"
    MacOS specific fix: The application often fails to find the 'cocoa' platform 
    plugin when running from a venv. We must manually point the environment 
    variable QT_QPA_PLATFORM_PLUGIN_PATH to the PyQt6 installation.
    \"\"\"
    if sys.platform == 'darwin':
        try:
            import PyQt6
            pkg_dir = os.path.dirname(PyQt6.__file__)
            
            # Standard pip install location: site-packages/PyQt6/Qt6/plugins
            plugin_path = os.path.join(pkg_dir, 'Qt6', 'plugins')
            
            # Fallback check
            if not os.path.exists(plugin_path):
                # Sometimes it's just inside site-packages/PyQt6/plugins
                plugin_path = os.path.join(pkg_dir, 'plugins')
            
            if os.path.exists(plugin_path):
                os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
                print(f"✅ Applied macOS Plugin Path: {plugin_path}")
            else:
                print("⚠️ Warning: Could not auto-detect PyQt6 plugin path.")
                
        except ImportError:
            pass

macos_plugin_fix()

# Imports must happen AFTER the fix
from PyQt6.QtWidgets import QApplication
from src.controller import AnnotationController
from src.model import SignalModel
from src.view import MainWindow
from src.style import APPLE_DARK_THEME

def main():
    # Universal High DPI support
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    
    app = QApplication(sys.argv)
    app.setStyleSheet(APPLE_DARK_THEME)
    
    model = SignalModel()
    view = MainWindow()
    controller = AnnotationController(model, view)
    
    view.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
""",

    "src/__init__.py": "",

    "src/style.py": """
APPLE_DARK_THEME = \"\"\"
QMainWindow { background-color: #1e1e1e; }
QWidget { font-family: 'SF Pro Display', 'Helvetica Neue', 'Helvetica', sans-serif; font-size: 13px; color: #e0e0e0; }
QToolBar { background-color: #2d2d2d; border-bottom: 1px solid #1a1a1a; padding: 8px; spacing: 10px; }
QToolButton { background-color: transparent; border: 1px solid transparent; border-radius: 4px; padding: 4px; color: #e0e0e0; }
QToolButton:hover { background-color: #3d3d3d; border: 1px solid #505050; }
QToolButton:pressed { background-color: #007AFF; color: white; }
QStatusBar { background-color: #2d2d2d; color: #808080; border-top: 1px solid #1a1a1a; }
QMessageBox { background-color: #2d2d2d; color: #e0e0e0; }
\"\"\"
""",

    "src/model.py": """import numpy as np
import os

class SignalModel:
    def __init__(self):
        self.data = None
        self.fs = 100 
        self.filepath = None
        self.filename = None
        self.n_channels = 0

    def load_from_file(self, filepath):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        raw_data = np.load(filepath, mmap_mode='r')
        
        if raw_data.ndim == 1:
            self.data = raw_data.reshape(1, -1)
        elif raw_data.ndim == 2:
            if raw_data.shape[0] < raw_data.shape[1]:
                self.data = raw_data
            else:
                self.data = raw_data.T
        
        self.n_channels = self.data.shape[0]

    def get_channel_data(self, channel_idx):
        if self.data is not None and channel_idx < self.n_channels:
            return self.data[channel_idx]
        return None
""",

    "src/view.py": """from PyQt6.QtWidgets import (QMainWindow, QVBoxLayout, QWidget, 
                             QToolBar, QStatusBar, QLabel, QMessageBox, QSizePolicy)
from PyQt6.QtGui import QAction
import pyqtgraph as pg

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PPG Annotator Pro (Mac)")
        self.resize(1200, 800)
        
        central = QWidget()
        self.setCentralWidget(central)
        self.layout = QVBoxLayout(central)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.plot_container = pg.GraphicsLayoutWidget()
        self.plot_container.setBackground('#1e1e1e')
        self.layout.addWidget(self.plot_container)
        
        self.plots = []
        self._setup_toolbar()
        self._setup_statusbar()

    def _setup_toolbar(self):
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        self.action_load = QAction("Load Signal", self)
        self.action_auto = QAction("Auto-Detect", self)
        self.action_save = QAction("Export Mask", self)
        self.action_add = QAction("Add Region (A)", self)
        self.action_del = QAction("Delete (Del)", self)
        
        self.action_add.setShortcut("A")
        self.action_del.setShortcut("Del")
        
        toolbar.addAction(self.action_load)
        toolbar.addSeparator()
        toolbar.addAction(self.action_auto)
        toolbar.addAction(self.action_add)
        toolbar.addAction(self.action_del)
        toolbar.addSeparator()
        toolbar.addAction(self.action_save)

    def _setup_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.lbl_info = QLabel("Ready")
        self.status_bar.addWidget(self.lbl_info)

    def create_channel_plots(self, n_channels):
        self.plot_container.clear()
        self.plots = []
        for i in range(n_channels):
            p = self.plot_container.addPlot(row=i, col=0)
            p.setLabel('left', f"Ch {i+1}")
            p.showGrid(x=True, y=True, alpha=0.2)
            p.setMouseEnabled(x=True, y=False)
            p.setDownsampling(mode='peak', auto=True)
            p.setClipToView(True)
            if i > 0:
                p.setXLink(self.plots[0])
            self.plots.append(p)

    def show_info(self, msg):
        QMessageBox.information(self, "Info", msg)
""",

    "src/controller.py": """import pyqtgraph as pg
from PyQt6.QtWidgets import QFileDialog
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import QApplication

class AnnotationController:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.regions = []

        self.view.action_load.triggered.connect(self.load_data)
        self.view.action_auto.triggered.connect(self.run_auto_detection)
        self.view.action_save.triggered.connect(self.export_binary_mask)
        self.view.action_add.triggered.connect(self.add_manual_region)
        self.view.action_del.triggered.connect(self.delete_region)

    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(self.view, "Load Data", "", "NumPy (*.npy)")
        if path:
            self.model.load_from_file(path)
            self.view.create_channel_plots(self.model.n_channels)
            self.render_all_channels()
            self.view.lbl_info.setText(f"Loaded {self.model.filename}")

    def render_all_channels(self):
        self.regions = []
        for i, plot_item in enumerate(self.view.plots):
            data = self.model.get_channel_data(i)
            color = ['#32d74b', '#0a84ff', '#ff9f0a'][i % 3]
            plot_item.plot(data, pen=pg.mkPen(color, width=1))

    def run_auto_detection(self):
        if self.model.data is None: return
        self.view.lbl_info.setText("Running detection...")
        QApplication.processEvents()
        
        signal = self.model.get_channel_data(0)
        series = pd.Series(signal)
        rolling_std = series.rolling(window=self.model.fs).std().fillna(0)
        threshold = rolling_std.median() * 3.0
        mask = rolling_std > threshold
        
        diff = np.diff(mask.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        self.clear_regions()
        count = 0
        for s, e in zip(starts, ends):
            if (e - s) > (0.5 * self.model.fs):
                self._create_region_item(s, e)
                count += 1
        self.view.lbl_info.setText(f"Auto-detected {count} regions.")

    def add_manual_region(self):
        if not self.view.plots: return
        vb = self.view.plots[0].getViewBox()
        x_min, x_max = vb.viewRange()[0]
        center = (x_min + x_max) / 2
        width = (x_max - x_min) * 0.1
        self._create_region_item(center - width/2, center + width/2)

    def _create_region_item(self, start, end):
        brush = pg.mkBrush(255, 69, 58, 80)
        hover_brush = pg.mkBrush(255, 69, 58, 120)
        region = pg.LinearRegionItem([start, end], brush=brush)
        region.setHoverBrush(hover_brush)
        if self.view.plots:
            self.view.plots[0].addItem(region)
            self.regions.append(region)

    def clear_regions(self):
        if self.view.plots:
            for r in self.regions:
                self.view.plots[0].removeItem(r)
        self.regions = []

    def delete_region(self):
        if self.regions:
            r = self.regions.pop()
            self.view.plots[0].removeItem(r)

    def export_binary_mask(self):
        if self.model.data is None: return
        n_samples = self.model.data.shape[1]
        mask = np.ones(n_samples, dtype=np.int8)
        for r in self.regions:
            min_x, max_x = r.getRegion()
            start = max(0, int(min_x))
            end = min(n_samples, int(max_x))
            mask[start:end] = 0
        save_name = self.model.filename.replace('.npy', '_mask.npy')
        save_path, _ = QFileDialog.getSaveFileName(self.view, "Save", save_name, "NumPy (*.npy)")
        if save_path:
            np.save(save_path, mask)
            self.view.show_info(f"Saved binary mask.")
"""
}

# Generate folders
if not os.path.exists(project_name):
    os.makedirs(project_name)
os.makedirs(os.path.join(project_name, "src"), exist_ok=True)

for filename, content in structure.items():
    with open(os.path.join(project_name, filename), "w") as f:
        f.write(content)

print(f"Project generated in folder: {project_name}")
