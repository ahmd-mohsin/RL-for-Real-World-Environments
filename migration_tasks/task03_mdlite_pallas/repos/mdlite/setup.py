"""Builds libmdlite.so with CMake (CUDA) and installs it inside the mdlite package."""
import os
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent


class BuildWithCMake(build_py):
    def run(self):
        super().run()
        build = ROOT / "build" / "cmake"
        build.mkdir(parents=True, exist_ok=True)
        jobs = os.environ.get("MAX_JOBS", str(os.cpu_count() or 4))
        subprocess.check_call(["cmake", "-S", str(ROOT), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"])
        subprocess.check_call(["cmake", "--build", str(build), "-j", jobs])
        target = Path(self.build_lib) / "mdlite"
        target.mkdir(parents=True, exist_ok=True)
        self.copy_file(str(build / "libmdlite.so"), str(target / "libmdlite.so"))
        if getattr(self, "editable_mode", False):
            self.copy_file(str(build / "libmdlite.so"), str(ROOT / "mdlite" / "libmdlite.so"))


setup(
    name="mdlite",
    version="1.3.0",
    packages=["mdlite"],
    python_requires=">=3.10",
    install_requires=["numpy"],
    cmdclass={"build_py": BuildWithCMake},
    entry_points={"console_scripts": ["mdlite = mdlite.cli:main"]},
    zip_safe=False,
)
