from setuptools import setup, find_packages

setup(
    name="SafeGym",
    version="0.1",
    packages=find_packages(),
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
)
