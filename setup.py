import pathlib
from setuptools import setup, find_packages

LICENSE: str = "MIT"
README: str = pathlib.Path("README.md").read_text()


setup(
    name="chalk-diagrams",
    version="0.1.2",
    packages=find_packages(
        include=["chalk"], 
        exclude=["examples", "docs", "test*"],
    ),
    description="A declarative drawing API",
    install_requires=[
        "pycairo",
        "toolz",
        "colour",
        "svgwrite",
        "cairosvg",
        "Pillow",
    ],
    extras_require={"latex": ["latextools"]},
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
