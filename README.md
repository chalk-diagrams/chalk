<p align="center"><img src="https://raw.githubusercontent.com/danoneata/chalk/master/examples/output/logo-sm.png" width=300></p>

Chalk is a declarative drawing library built on top of [PyCairo](https://pycairo.readthedocs.io).
The API draws heavy inspiration from
Haskell's [diagrams](https://diagrams.github.io/),
Scala's [doodle](https://github.com/creativescala/doodle/) and
Jeremy Gibbons's lecture notes on [Functional Programming for Domain−Specific Languages](http://www.cs.ox.ac.uk/publications/publication7583-abstract.html).

⚠️ The library is still very much work in progress and subject to change.

# Installation

The library is available on PyPI as `chalk-diagrams` and can be installed with `pip`:
```bash
pip install chalk-diagrams
```
On Debian (or Colab) you will need to install Cairo

```bash
sudo apt-get install libcairo2-dev
```

If you want to use the LaTeX extension, run:
```bash
pip install chalk-diagrams[latex]
```
For the LaTeX extension you might need to install `pdf2svg` and `texlive`;
on Debian these dependencies can be installed as follows:
```bash
sudo apt-get install pdf2svg texlive texlive-science texlive-latex-recommended texlive-latex-extra
```

## Overview

Below we provide a brief introduction of the main functionality of the library.
These examples are available in the `examples/intro.py` file.

We start by importing the [`colour`](https://github.com/vaab/colour) module and the `diagrams` functions:


```python
from colour import Color
from chalk import *
```

We also define some colors that will be shortly used:


```python
papaya = Color("#ff9700")
blue = Color("#005FDB")
```

We can easily create basic shapes (the functions `circle`, `square`, `triangle`) and style them with various attributes (the methods`fill_color`, `line_color`, `line_width`).
For example:


```python
d = circle(1).fill_color(papaya)
d
```




    
![svg](README_files/README_14_0.svg)
    



The diagram can be saved to an image using the `render` method:


```python
d.render("examples/output/intro-01.png", height=64)
```

We can glue together two diagrams using the combinators `atop` (or `+`), `beside` (or `|`), `above` (or `/`).
For example:


```python
circle(0.5).fill_color(papaya) | square(1).fill_color(blue)
```




    
![svg](README_files/README_18_0.svg)
    



which is equivalent to


```python
circle(0.5).fill_color(papaya).beside(square(1).fill_color(blue))
```




    
![svg](README_files/README_20_0.svg)
    



We also provide combinators for a list of diagrams:
`hcat` for horizontal composition, `vcat` for vertical composition.
For example:


```python
hcat(circle(0.1 * i) for i in range(1, 6)).fill_color(blue)
```




    
![svg](README_files/README_22_0.svg)
    



We can use Python functions to build more intricate diagrams:


```python
def sierpinski(n: int, size: int) -> Diagram:
    if n <= 1:
        return triangle(size)
    else:
        smaller = sierpinski(n - 1, size / 2)
        return smaller.above(smaller.beside(smaller).center_xy())
```


```python
sierpinski(5, 4).fill_color(papaya)
```




    
![svg](README_files/README_25_0.svg)
    



For more examples, please check the `examples` folder.
These scripts can be run as follows:

```bash
python examples/squares.py
```

![squares](https://raw.githubusercontent.com/danoneata/chalk/master/examples/output/squares.png)

```bash
python examples/escher_square_limit.py
```

![escher](https://raw.githubusercontent.com/danoneata/chalk/master/examples/output/escher-square-limit.png)

```bash
python examples/hanoi.py
```

![hanoi](https://raw.githubusercontent.com/danoneata/chalk/master/examples/output/hanoi.png)

## Authors

- [Dan Oneață](http://doneata.bitbucket.io/)
- [Alexander Rush](http://rush-nlp.com/)

Special thanks to [Ionuț G. Stan](http://igstan.ro/) for providing many useful insights and comments!
