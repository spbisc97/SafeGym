from setuptools import setup, find_packages

# import numpy
# from Cython.Build import cythonize
# from distutils.extension import Extension
# ext_modules = [Extension("*",["./safegym/envs/*.py"],include_dirs=[numpy.get_include()])]


import pathlib

CWD = pathlib.Path(__file__).absolute().parent


setup(
    name="safegym",
    version="0.12",
    include_dirs=["safegym", "safegym.*"],
    install_requires=[
        "numpy>=1.23",
        "gymnasium",
    ],
    # Metadata
    author="Simone Rotondi",
    author_email="rotondi97simone@gmail.com",
    description="Gymnasium Compatible Safe RL Environments",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/spbisc97/SafeGym",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    # setup_requires=['cython',
    #               'setuptools'],
    # ext_modules=cythonize(
    # ext_modules,
    # compiler_directives={'language_level' : "3"}),
)
