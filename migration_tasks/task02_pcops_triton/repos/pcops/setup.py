import glob
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

here = os.path.dirname(os.path.abspath(__file__))
sources = ["csrc/pcops.cpp"] + sorted(
    os.path.relpath(p, here) for p in glob.glob(os.path.join(here, "csrc", "**", "*.cu"), recursive=True)
)

setup(
    name="pcops",
    version="0.4.0",
    packages=["pcops"],
    ext_modules=[
        CUDAExtension(
            "pcops._C",
            sources,
            include_dirs=[os.path.join(here, "csrc")],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--expt-relaxed-constexpr"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
