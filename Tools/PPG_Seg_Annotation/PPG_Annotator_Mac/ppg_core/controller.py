import pyqtgraph as pg
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
