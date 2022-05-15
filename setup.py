from distutils.core import setup


with open("README.md", "r") as f:
    README = f.read()


setup(
    name="chalk",
    version="0.1.dev",
    packages=[
        "chalk",
    ],
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
)
