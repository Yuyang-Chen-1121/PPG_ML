
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
