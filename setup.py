from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# 定义Cython扩展 - 编译pure_math.py
extensions = [
    Extension(
        "pure_math",
        ["pure_math.py"],
        include_dirs=[np.get_include()],
        language_level="3",
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        extra_compile_args=['/O2'] if os.name == 'nt' else ['-O3'],
    )
]

# 编译选项
compiler_directives = {
    'language_level': "3",
    'boundscheck': False,
    'wraparound': False,
    'initializedcheck': False,
    'cdivision': True,
    'infer_types': True,
    'embedsignature': True,
}

setup(
    name="pure_math",
    ext_modules=cythonize(
        extensions,
        compiler_directives=compiler_directives,
        annotate=True,  # 生成HTML注释文件，用于查看性能瓶颈
    ),
    zip_safe=False,
)