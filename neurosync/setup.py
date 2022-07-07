from setuptools import setup, Extension, Command
from Cython.Build import cythonize
import numpy

compiler_directives = {"language_level": 3}
compiler_args=['-Wno-cpp']

ext_modules=[
    Extension(name="EnumList", sources=["EnumList.pyx"], extra_compile_args=compiler_args),
    Extension(name="Neuron", sources=["Neuron.pyx"], extra_compile_args=compiler_args),
    Extension(name="Checkpoint", sources=["Checkpoint.pyx"], extra_compile_args=compiler_args),
    Extension(name="RRManager", sources=["RRManager.pyx"], extra_compile_args=compiler_args),
    Extension(name="Router", sources=["Router.pyx"], extra_compile_args=compiler_args),
    Extension(name="Core", sources=["Core.pyx"], extra_compile_args=compiler_args),
    Extension(name="NoC", sources=["NoC.pyx"], extra_compile_args=compiler_args),
    Extension(name="Init", sources=["Init.pyx"], extra_compile_args=compiler_args),
    Extension(name="Learning", sources=["Learning.pyx"], extra_compile_args=compiler_args)
]

setup(
    name = 'NeuroSync',
    ext_modules = cythonize(ext_modules, compiler_directives=compiler_directives),
    include_dirs=[numpy.get_include()]
)
