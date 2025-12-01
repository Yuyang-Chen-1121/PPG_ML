
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
