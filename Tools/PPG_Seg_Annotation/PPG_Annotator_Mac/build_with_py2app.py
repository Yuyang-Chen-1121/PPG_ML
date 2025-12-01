
import os
import subprocess
import sys
import shutil

def build():
    print("===================================")
    print("   PPG Annotator - Clean Venv Builder")
    print("===================================")

    # 1. Define Venv Path
    # We create a specific build environment to isolate from Anaconda bloat
    venv_dir = os.path.join(os.getcwd(), "build_venv")
    venv_python = os.path.join(venv_dir, "bin", "python")

    # 2. Create Clean Virtual Environment
    print(f"🧹 Creating fresh virtual environment at {venv_dir}...")
    if os.path.exists(venv_dir):
        shutil.rmtree(venv_dir)
    
    # Use the current python to create the venv
    subprocess.check_call([sys.executable, "-m", "venv", "build_venv"])

    # 3. Install Minimal Dependencies
    print("📦 Installing minimal dependencies into build_venv...")
    # Crucial: Only install what is needed. No jupyter, no zmq.
    pkgs = ["PyQt6", "pyqtgraph", "numpy", "pandas", "scipy", "py2app"]
    
    # Install (Upgrade pip first)
    subprocess.check_call([venv_python, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([venv_python, "-m", "pip", "install"] + pkgs)

    # 4. Clean previous builds
    print("🧹 Cleaning old build artifacts...")
    for folder in ["dist", "build", ".eggs"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    
    # 5. Create/Ensure setup.py exists
    if not os.path.exists("setup.py"):
        print("❌ Error: setup.py not found. Please run fix_bug.py first.")
        return

    # 6. Run Build Command using the VENV Python
    print("🚀 Building .app package using isolated environment...")
    cmd = [venv_python, "setup.py", "py2app"]
    
    try:
        subprocess.check_call(cmd)
        print("\n✅ Build Complete!")
        print(f"👉 Your app is ready at: {os.path.abspath('dist/PPG Annotator.app')}")
        print("   (You can drag this to your Applications folder)")
        print("   (You can delete the 'build_venv' folder now if you want)")
    except subprocess.CalledProcessError:
        print("\n❌ Build Failed.")
        print("Please check the error logs above.")

if __name__ == "__main__":
    build()
