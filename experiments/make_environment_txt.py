"""Record the exact environment this project ran in, so results can be
reproduced or sanity-checked later. Writes results/environment.txt."""

import datetime
import platform
import subprocess
import sys


def get_package_versions():
    packages = ["numpy", "scipy", "scikit-learn", "scikit-image",
                "opencv-python-headless", "matplotlib", "pandas",
                "torch", "torchvision", "tqdm", "pytest", "pyyaml"]
    lines = []
    for pkg in packages:
        try:
            mod_name = {
                "scikit-learn": "sklearn",
                "scikit-image": "skimage",
                "opencv-python-headless": "cv2",
                "pyyaml": "yaml",
            }.get(pkg, pkg)
            mod = __import__(mod_name)
            version = getattr(mod, "__version__", "unknown")
            lines.append(f"{pkg}=={version}")
        except ImportError:
            lines.append(f"{pkg}==NOT INSTALLED")
    return lines


def get_cpu_model():
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        return "unknown (sysctl failed, not on macOS?)"


def get_ram_gb():
    try:
        total_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        return f"{total_bytes / (1024 ** 3):.1f} GB"
    except Exception:
        return "unknown"


def main():
    lines = []
    lines.append(f"Date: {datetime.datetime.now().isoformat()}")
    lines.append(f"Python: {sys.version}")
    lines.append(f"Platform: {platform.platform()}")
    lines.append(f"CPU: {get_cpu_model()}")
    lines.append(f"RAM: {get_ram_gb()}")
    lines.append("")
    lines.append("Package versions:")
    lines += get_package_versions()

    text = "\n".join(lines) + "\n"
    with open("results/environment.txt", "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
