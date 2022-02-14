An exploration into building a declarative API for drawing over PyCairo.
The design draws inspiration from the [diagrams](https://diagrams.github.io/) library in Haskell.

## Overview

This section gives an overview of the main functionality of the library.
These examples are also available in the file `examples/intro.py`.

We start by importing the color module and the diagrams functions:

```python
from colour import Color
from diagrams import *
```

Shapes and attributes:

```python
papaya = Color("#ff9700")
d = circle(1).fill_color(papaya)
```
![circle](https://github.com/danoneata/pydiagrams/blob/master/examples/output/intro-01.png)

Combining diagrams:

```python
blue = Color("#005FDB")
circle(2).fill_color(papaya) | square(1).fill_color(blue)
```

which is equivalent to

```python
circle(2).fill_color(papaya).beside(square(1).fill_color(blue))
```

![atop](https://github.com/danoneata/pydiagrams/blob/master/examples/output/intro-02.png)

Combining diagrams, horizontal composition:

```python
hcat(circle(0.1 * i) for i in range(1, 6)).fill_color(blue)
```
![hcat](https://github.com/danoneata/pydiagrams/blob/master/examples/output/intro-03.png)

## More examples

```python
streamlit run examples/squares.py
```

![squares](https://github.com/danoneata/pydiagrams/blob/master/examples/output/squares.png)

```python
streamlit run examples/escher_square_limit.py
```

![escher](https://github.com/danoneata/pydiagrams/blob/master/examples/output/escher_square_limit.png)

## TODO

- [ ] Draw paths
- [ ] Allow change of file type for the `render` function
- [ ] Add `rotate_by` transformation (rotates by a fraction of the circle)
- [ ] Define color names
- [ ] Add more examples: e.g., Vera Molnár's drawings, Hilbert curve, Escher's Square Limit
- [ ] How to allow the context?
- [x] Allow for HSV colors
- [x] Composing transformations is not efficient.
- [x] Transform bounding boxes
- [x] Add `Scale` transformation
- [x] Show origin helper function
- [x] Update width and height of image to fit all the contents
