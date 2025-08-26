from setuptools import setup, find_packages

# import numpy
# from Cython.Build import cythonize
# from distutils.extension import Extension
# ext_modules = [Extension("*",["./safegym/envs/*.py"],include_dirs=[numpy.get_include()])]


import pathlib
import re

CWD = pathlib.Path(__file__).absolute().parent

# Read install requirements from local requirements.txt
def _read_requirements(filename: str = "requirements.txt"):
    req_path = CWD / filename
    if not req_path.exists():
        return []
    lines = []
    with open(req_path, "r", encoding="utf-8") as f:
        for raw in f.readlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
    return lines

# Optional safety check: if moviepy is already installed but < 2, raise
def _assert_moviepy_v2_if_installed():
    try:
        import importlib
        mp = importlib.import_module("moviepy")
        ver = getattr(mp, "__version__", "0")
        m = re.match(r"(\d+)\.(\d+)", ver or "0")
        major = int(m.group(1)) if m else 0
        if major < 2:
            raise RuntimeError(
                f"Detected moviepy {ver}. SafeGym requires moviepy>=2. "
                "Please upgrade: pip install -U moviepy"
            )
    except ModuleNotFoundError:
        # Not installed yet; pip will resolve via install_requires
        pass
    except Exception:
        # Do not hard fail on unexpected environments
        pass

_assert_moviepy_v2_if_installed()


setup(
    name="safegym",
    version="0.18",
    include_dirs=["safegym", "safegym.*"],
    install_requires=_read_requirements(),
    extras_require={
        # Optional environments / tools
        "pygame": ["pygame"],  # SnakeEnv
        "sb3": ["stable-baselines3>=2"],  # examples and notebooks
        # Mujoco-based env support
        "mujoco": [
            "gymnasium[mujoco]>=1.0",
            "mujoco",  # Python mujoco runtime
        ],
        # Everything
        "all": [
            "pygame",
            "stable-baselines3>=2",
            "gymnasium[mujoco]>=1.0",
            "mujoco",
        ],
    },
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
