import pathlib

from setuptools import find_packages, setup

LICENSE: str = "MIT"
README: str = pathlib.Path("README.md").read_text(encoding="utf-8")

# ---------------------------------------------------------------
# NOTE:
# Since the library name (chalk-diagrams) is different from
#   the module (chalk), we set the custom dunder attribute
#   __libname__ in chalk/__init__.py and use it there to fetch
#   and set __version__ with library metadata inside
#   chalk/__init__.py.
#   Since, library name will not be changed in future, it is
#   being maintained at two places
#   1. setup.py
#   1. chalk/__init__.py
#
#   The version will be updated often only from setup.py.
# ---------------------------------------------------------------
LIBNAME: str = "chalk-diagrams"

setup(
    name=LIBNAME,
    version="0.1.2",
    packages=find_packages(
        include=["chalk"],
        exclude=["examples", "docs", "test*"],
    ),
    description="A declarative drawing API",
    install_requires=[
        "toolz",
        "colour",
        "svgwrite",
        "Pillow",
        "loguru",
        "chalk-planar",
        "typing-extensions",
    ],
    extras_require={
        "tikz": ["pylatex"],
        "latex": ["latextools"],
        "png": ["pycairo"],
        "svg": ["cairosvg"],
    },
    long_description=README,
    long_description_content_type="text/markdown",
    author="Dan Oneață",
    author_email="dan.oneata@gmail.com",
    url="https://github.com/danoneata/chalk",
    project_urls={
        "Documentation": "https://github.com/danoneata/chalk",
        "Source Code": "https://github.com/danoneata/chalk",
        "Issue Tracker": "https://github.com/danoneata/chalk/issues",
    },
    license=LICENSE,
    license_files=("LICENSE",),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        f"License :: OSI Approved :: {LICENSE} License",
        "Topic :: Scientific/Engineering",
    ],
)
