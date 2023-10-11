from setuptools import setup, find_packages

# import numpy
# from Cython.Build import cythonize
# from distutils.extension import Extension
# ext_modules = [Extension("*",["./safegym/envs/*.py"],include_dirs=[numpy.get_include()])]


setup(
    name="SafeGym",
    version="0.1",
    packages=find_packages(),
    package_data={"safegym": ["typed.py"]},
    install_requires=[
        "numpy>=1.10",
        "gymnasium",
    ],
    # Metadata
    author="Simone Rotondi",
    # author_email=,
    # description=,
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
