import os
import shutil

# ==========================================
# 0. CLEANUP (Remove old 'src' to avoid confusion)
# ==========================================
if os.path.exists("src"):
    print("♻️  Refactoring: Removing old 'src' folder to switch to 'ppg_core'...")
    shutil.rmtree("src")

# ==========================================
# 1. MAIN.PY (Updated imports for ppg_core)
# ==========================================
main_code = """
import sys
import os

# Smart Plugin Fix for Development Mode
if not getattr(sys, 'frozen', False):
    if sys.platform == 'darwin':
        try:
            import PyQt6
            pkg_dir = os.path.dirname(PyQt6.__file__)
            plugin_path = os.path.join(pkg_dir, 'Qt6', 'plugins')
            if not os.path.exists(plugin_path):
                plugin_path = os.path.join(pkg_dir, 'plugins')
            if os.path.exists(plugin_path):
                os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
        except ImportError:
            pass

from PyQt6.QtWidgets import QApplication
from ppg_core.controller import AnnotationController
from ppg_core.model import SignalModel
from ppg_core.view import MainWindow
from ppg_core.style import APPLE_DARK_THEME

def main():
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
"""

# ==========================================
# 2. STYLE.PY (Themes)
# ==========================================
style_code = """
APPLE_DARK_THEME = \"\"\"
QMainWindow { background-color: #1e1e1e; }
QWidget { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 13px; color: #e0e0e0; }
QToolBar { background-color: #2d2d2d; border-bottom: 1px solid #1a1a1a; padding: 8px; spacing: 10px; }
QToolButton { background-color: transparent; border: 1px solid transparent; border-radius: 4px; padding: 4px; color: #e0e0e0; }
QToolButton:hover { background-color: #3d3d3d; border: 1px solid #505050; }
QToolButton:pressed { background-color: #007AFF; color: white; }
QMenu { background-color: #2d2d2d; border: 1px solid #454545; }
QMenu::item { padding: 5px 20px; background-color: transparent; color: #e0e0e0; }
QMenu::item:selected { background-color: #007AFF; color: white; }
QStatusBar { background-color: #2d2d2d; color: #808080; border-top: 1px solid #1a1a1a; }
QMessageBox { background-color: #2d2d2d; color: #e0e0e0; }
QDialog { background-color: #2d2d2d; }
QLabel { color: #e0e0e0; }
QLineEdit { background-color: #1e1e1e; border: 1px solid #3a3a3a; color: #e0e0e0; padding: 4px; }
QDialogButtonBox QPushButton { background-color: #007AFF; color: white; border-radius: 4px; padding: 6px 16px; border: none; }
\"\"\"

APPLE_LIGHT_THEME = \"\"\"
QMainWindow { background-color: #f5f5f7; }
QWidget { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 13px; color: #1d1d1f; }
QToolBar { background-color: #ffffff; border-bottom: 1px solid #d2d2d7; padding: 8px; spacing: 10px; }
QToolButton { background-color: transparent; border: 1px solid transparent; border-radius: 4px; padding: 4px; color: #1d1d1f; }
QToolButton:hover { background-color: #e5e5ea; border: 1px solid #d2d2d7; }
QToolButton:pressed { background-color: #007AFF; color: white; }
QMenu { background-color: #ffffff; border: 1px solid #d2d2d7; }
QMenu::item { padding: 5px 20px; background-color: transparent; color: #1d1d1f; }
QMenu::item:selected { background-color: #007AFF; color: white; }
QStatusBar { background-color: #f5f5f7; color: #86868b; border-top: 1px solid #d2d2d7; }
QMessageBox { background-color: #ffffff; color: #1d1d1f; }
QDialog { background-color: #ffffff; }
QLabel { color: #1d1d1f; }
QLineEdit { background-color: #ffffff; border: 1px solid #d2d2d7; color: #1d1d1f; padding: 4px; }
QDialogButtonBox QPushButton { background-color: #007AFF; color: white; border-radius: 4px; padding: 6px 16px; border: none; }
\"\"\"
"""

# ==========================================
# 3. MODEL.PY (Logic with Windowing)
# ==========================================
model_code = """import numpy as np
import os

class SignalModel:
    def __init__(self):
        self.data = None
        self.fs = 100 
        self.filepath = None
        self.filename = None
        self.is_segmented = False
        self.original_shape = None
        self.current_win_size = 256
        self.current_stride = 128

    def load_from_file(self, filepath, win_size=None, stride=None):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        
        try:
            raw_data = np.load(filepath, mmap_mode='r')
        except Exception as e:
            raise e

        self.original_shape = raw_data.shape
        
        # Segmented Data Logic
        if raw_data.ndim >= 2:
            self.is_segmented = True
            
            # Determine params (User input > Inferred > Default)
            _win = win_size if win_size else raw_data.shape[-1]
            _stride = stride if stride else (_win // 2)
            
            self.current_win_size = _win
            self.current_stride = _stride

            # Reconstruct
            segments = raw_data.reshape(-1, _win)
            if segments.shape[0] > 0:
                body = segments[:-1, :_stride].flatten()
                tail = segments[-1, :].flatten()
                self.data = np.concatenate([body, tail])
            else:
                self.data = segments.flatten()
            self.data = self.data.astype(np.float32)
        else:
            self.is_segmented = False
            self.data = raw_data.astype(np.float32).reshape(-1)

    def get_data(self):
        return self.data
"""

# ==========================================
# 4. VIEW.PY (With Dialog)
# ==========================================
view_code = """from PyQt6.QtWidgets import (QMainWindow, QVBoxLayout, QWidget, 
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
"""

# ==========================================
# 5. CONTROLLER.PY (Logic)
# ==========================================
controller_code = """import pyqtgraph as pg
from PyQt6.QtWidgets import QFileDialog, QApplication, QDialog
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt, QEvent, QObject
import numpy as np
import pandas as pd
from ppg_core.style import APPLE_DARK_THEME, APPLE_LIGHT_THEME
from ppg_core.view import SegmentationDialog

class AnnotationController(QObject):
    def __init__(self, model, view):
        super().__init__()
        self.model = model
        self.view = view
        self.is_dark_mode = True
        self.auto_mask = None 
        self.main_curve = None
        self.artifact_curve = None
        self.mode = 'NAV'
        self.drag_start = None
        self.temp_region = None

        self.view.action_load.triggered.connect(self.load_data)
        self.view.action_theme.triggered.connect(self.toggle_theme)
        self.view.action_auto.triggered.connect(self.run_auto_detection)
        self.view.action_clear_auto.triggered.connect(self.clear_all_masks)
        self.view.action_save.triggered.connect(self.export_binary_mask)
        self.view.action_nav.triggered.connect(lambda: self.set_mode('NAV'))
        self.view.action_paint.triggered.connect(lambda: self.set_mode('PAINT'))
        self.view.action_erase.triggered.connect(lambda: self.set_mode('ERASE'))
        self.view.plot_widget.plotItem.scene().installEventFilter(self)

    def toggle_theme(self):
        self.is_dark_mode = not self.is_dark_mode
        app = QApplication.instance()
        if self.is_dark_mode:
            app.setStyleSheet(APPLE_DARK_THEME)
            self.view.apply_plot_theme(dark=True)
        else:
            app.setStyleSheet(APPLE_LIGHT_THEME)
            self.view.apply_plot_theme(dark=False)
        if self.model.data is not None:
            self.render_base_plot()
            self.update_artifact_visuals()

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'NAV':
            self.view.set_cursor(Qt.CursorShape.ArrowCursor)
            self.view.plot_widget.setMouseEnabled(x=True, y=True)
            self.view.lbl_info.setText("Mode: Navigation")
        else:
            self.view.set_cursor(Qt.CursorShape.CrossCursor)
            self.view.plot_widget.setMouseEnabled(x=False, y=False)
            self.view.lbl_info.setText(f"Mode: {mode}")

    def eventFilter(self, source, event):
        if self.mode == 'NAV' or self.model.data is None: return False
        if event.type() == QEvent.Type.GraphicsSceneMousePress and event.button() == Qt.MouseButton.LeftButton:
            self.start_drag(event)
            return True
        elif event.type() == QEvent.Type.GraphicsSceneMouseMove and self.drag_start is not None:
            self.update_drag(event)
            return True
        elif event.type() == QEvent.Type.GraphicsSceneMouseRelease and self.drag_start is not None:
            self.finish_drag(event)
            return True
        return False

    def start_drag(self, event):
        pos = event.scenePos()
        mouse_point = self.view.plot_widget.plotItem.vb.mapSceneToView(pos)
        self.drag_start = mouse_point.x()
        brush_color = (255, 69, 58, 50) if self.mode == 'PAINT' else (50, 215, 75, 50)
        self.temp_region = pg.LinearRegionItem([self.drag_start, self.drag_start], brush=pg.mkBrush(brush_color))
        self.temp_region.setMovable(False)
        self.view.plot_widget.addItem(self.temp_region)

    def update_drag(self, event):
        pos = event.scenePos()
        mouse_point = self.view.plot_widget.plotItem.vb.mapSceneToView(pos)
        self.temp_region.setRegion([self.drag_start, mouse_point.x()])

    def finish_drag(self, event):
        pos = event.scenePos()
        mouse_point = self.view.plot_widget.plotItem.vb.mapSceneToView(pos)
        drag_end = mouse_point.x()
        self.view.plot_widget.removeItem(self.temp_region)
        self.temp_region = None
        self.apply_mask_range(min(self.drag_start, drag_end), max(self.drag_start, drag_end), is_artifact=(self.mode == 'PAINT'))
        self.drag_start = None

    def apply_mask_range(self, start_x, end_x, is_artifact):
        if self.auto_mask is None: return
        start_idx = max(0, int(start_x))
        end_idx = min(len(self.auto_mask), int(end_x))
        if end_idx > start_idx:
            self.auto_mask[start_idx:end_idx] = is_artifact
            self.update_artifact_visuals()

    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(self.view, "Load Data", "", "NumPy (*.npy)")
        if not path: return
        
        # Check if segmented
        win_size, stride = None, None
        try:
            temp = np.load(path, mmap_mode='r')
            if temp.ndim >= 2:
                dialog = SegmentationDialog(self.view, 256, 128)
                if dialog.exec() == QDialog.DialogCode.Accepted:
                    win_size, stride = dialog.get_values()
                else:
                    return
            del temp
        except: pass

        try:
            self.model.load_from_file(path, win_size, stride)
            self.auto_mask = np.zeros(len(self.model.data), dtype=bool)
            self.render_base_plot()
            self.view.lbl_info.setText(f"Loaded {self.model.filename}")
        except Exception as e:
            self.view.show_info(f"Load Error: {e}")

    def render_base_plot(self):
        self.view.plot_widget.clear()
        if self.model.data is not None:
            c = '#32d74b' if self.is_dark_mode else '#007AFF'
            self.main_curve = self.view.plot_widget.plot(self.model.data, pen=pg.mkPen(c, width=1))
            self.artifact_curve = self.view.plot_widget.plot([], [], pen=pg.mkPen('#ff453a', width=1), connect='finite')

    def run_auto_detection(self):
        if self.model.data is None: return
        self.view.lbl_info.setText("Detecting...")
        QApplication.processEvents()
        signal = self.model.get_data()
        series = pd.Series(signal)
        rolling_std = series.rolling(window=self.model.fs).std().fillna(0)
        threshold = rolling_std.median() * 3.0
        self.auto_mask = (rolling_std > threshold).to_numpy()
        self.update_artifact_visuals()
        self.view.lbl_info.setText("Detection Complete.")

    def clear_all_masks(self):
        if self.model.data is None: return
        self.auto_mask.fill(False)
        self.update_artifact_visuals()

    def update_artifact_visuals(self):
        if self.model.data is None: return
        display = self.model.data.copy()
        display[~self.auto_mask] = np.nan
        self.artifact_curve.setData(display)

    def export_binary_mask(self):
        if self.model.data is None: return
        n = len(self.model.data)
        mask_1d = np.ones(n, dtype=np.int8)
        mask_1d[self.auto_mask] = 0
        
        save_path, _ = QFileDialog.getSaveFileName(self.view, "Save", self.model.filename.replace('.npy', '_mask.npy'), "NumPy (*.npy)")
        if not save_path: return

        # Ask for segmentation setting
        dialog = SegmentationDialog(self.view, self.model.current_win_size, self.model.current_stride)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            win, stride = dialog.get_values()
            try:
                # Re-segment
                segments = []
                for i in range(0, len(mask_1d) - win + 1, stride):
                    segments.append(mask_1d[i : i + win])
                
                seg_mask = np.array(segments)
                # Restore original shape dims if needed
                if self.model.original_shape and len(self.model.original_shape) == 3:
                    seg_mask = seg_mask.reshape(seg_mask.shape[0], 1, win)
                
                np.save(save_path, seg_mask)
                self.view.show_info(f"Saved Segmented: {seg_mask.shape}")
            except Exception as e:
                self.view.show_info(f"Export Error: {e}")
"""

# ==========================================
# 6. SETUP.PY (Clean & Simple)
# ==========================================
setup_py_code = """
import sys
import os
# Recursion limit fix for deep dependencies
sys.setrecursionlimit(5000)

from setuptools import setup

APP = ['main.py']
DATA_FILES = ['icon.icns']

# --- LIBFFI FIX (Updated for Anaconda 3.12 compatibility) ---
# We scan multiple locations to find the correct libffi.8.dylib
libffi_found = None
possible_paths = [
    # Check current Python env first
    os.path.join(sys.base_prefix, 'lib', 'libffi.8.dylib'),
    os.path.join(sys.prefix, 'lib', 'libffi.8.dylib'),
    # Common Anaconda paths
    '/opt/anaconda3/lib/libffi.8.dylib',
    os.path.expanduser('~/anaconda3/lib/libffi.8.dylib'),
    os.path.expanduser('~/opt/anaconda3/lib/libffi.8.dylib'),
    # System/Homebrew fallbacks
    '/usr/local/lib/libffi.8.dylib',
    '/opt/homebrew/lib/libffi.8.dylib',
    '/opt/homebrew/opt/libffi/lib/libffi.8.dylib',
    '/usr/lib/libffi.8.dylib'
]

frameworks_list = []
for p in possible_paths:
    if os.path.exists(p):
        print(f"✅ FOUND libffi at: {p}")
        frameworks_list.append(p)
        libffi_found = p
        break

if not libffi_found:
    print("⚠️ WARNING: Could not find libffi.8.dylib. App may crash.")

OPTIONS = {
    'argv_emulation': False,
    'iconfile': 'icon.icns',
    'includes': ['PyQt6', 'pyqtgraph', 'numpy', 'pandas', 'scipy.signal', 'ctypes'],
    'packages': ['PyQt6', 'pyqtgraph', 'pandas', 'numpy', 'scipy'],
    'frameworks': frameworks_list,
    'excludes': ['PyQt5', 'tkinter', 'matplotlib', 'PyInstaller', 'PySide6', 'PySide2', 'zmq', 'IPython', 'jupyter'],
    'plist': {
        'CFBundleName': 'PPG Annotator',
        'CFBundleDisplayName': 'PPG Annotator',
        'CFBundleGetInfoString': "PPG Annotation Tool",
        'CFBundleIdentifier': "com.ppg.annotator.pro",
        'CFBundleVersion': "1.2.0",
        'CFBundleShortVersionString': "1.2.0",
        'NSHighResolutionCapable': True
    }
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
"""

# ==========================================
# WRITE FILES
# ==========================================
# Create new package folder
base_dir = "ppg_core"
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# Helper to write files
def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# Write Source Files
write_file(os.path.join(base_dir, "__init__.py"), "")
write_file(os.path.join(base_dir, "style.py"), style_code)
write_file(os.path.join(base_dir, "model.py"), model_code)
write_file(os.path.join(base_dir, "view.py"), view_code)
write_file(os.path.join(base_dir, "controller.py"), controller_code)

# Write Root Files
write_file("main.py", main_code)
write_file("setup.py", setup_py_code)

print("\n✅ Code Refactored: 'src' renamed to 'ppg_core' to fix packaging errors.")
print("✅ New Features Added: Segmentation Dialogs.")
print("👉 Run 'python3 build_with_py2app.py' to build the App.")
