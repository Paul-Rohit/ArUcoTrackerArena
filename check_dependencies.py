"""
BotArena Dependency & Environment Checker
File: check_dependencies.py

Checks for required packages:
- mujoco
- opencv-contrib-python (with aruco support check)
- numpy
- matplotlib

Generates or updates 'requirements.txt' containing ONLY missing libraries.
"""

import importlib
import os
import sys

# Mapping: import module name -> PyPI distribution package name
REQUIRED_DEPS = {
    "mujoco": "mujoco>=3.0.0",
    "cv2": "opencv-contrib-python>=4.8.0.76",
    "numpy": "numpy>=1.24.0",
    "matplotlib": "matplotlib>=3.7.0",
}

REQUIREMENTS_FILE = "requirements.txt"


def check_environment():
    missing_packages = []
    installed_packages = {}

    print("=" * 65)
    print(" BOTARENA DEPENDENCY & RUNTIME VERIFICATION")
    print(f" Python Executable: {sys.executable}")
    print(f" Python Version   : {sys.version.split()[0]}")
    print("=" * 65)

    for mod_name, pip_pkg in REQUIRED_DEPS.items():
        try:
            mod = importlib.import_module(mod_name)
            version = getattr(mod, "__version__", "installed (no __version__)")

            # Specialized check: OpenCV must include the aruco submodule
            if mod_name == "cv2":
                if not hasattr(mod, "aruco"):
                    print(
                        f"[-] [FAIL] cv2 is installed ({version}), but 'cv2.aruco' is missing!"
                    )
                    print(
                        "           (You likely have headless/standard 'opencv-python' instead of 'opencv-contrib-python')"
                    )
                    missing_packages.append(pip_pkg)
                    continue

            print(f"[+] [OK]   {mod_name:<12} -> Version {version}")
            installed_packages[mod_name] = version

        except ImportError:
            print(f"[-] [FAIL] {mod_name:<12} -> NOT FOUND")
            missing_packages.append(pip_pkg)

    print("-" * 65)

    # If everything is installed
    if not missing_packages:
        print("[SUCCESS] All required dependencies are present and operational.")
        # Remove old requirements.txt if previously left behind
        if os.path.exists(REQUIREMENTS_FILE):
            try:
                os.remove(REQUIREMENTS_FILE)
            except OSError:
                pass
        sys.exit(0)

    # If any package is missing, generate targeted requirements.txt
    print(f"[!] Detected {len(missing_packages)} missing or incomplete package(s).")
    print(f"[!] Generating clean '{REQUIREMENTS_FILE}' for missing modules...")

    with open(REQUIREMENTS_FILE, "w", encoding="utf-8") as f:
        f.write("# Generated automatically by check_dependencies.py\n")
        f.write("# BotArena missing dependencies:\n")
        for pkg in missing_packages:
            f.write(f"{pkg}\n")

    print(f"[+] '{REQUIREMENTS_FILE}' successfully created with contents:")
    for pkg in missing_packages:
        print(f"    - {pkg}")

    print("\nTo install the missing packages, run:")
    print(f"    pip install -r {REQUIREMENTS_FILE}")
    print("=" * 65)

    sys.exit(1)


if __name__ == "__main__":
    check_environment()