An exploration into building a declarative API for drawing over PyCairo.
The design draws inspiration from the [diagrams](https://diagrams.github.io/) library in Haskell.

## Overview

Shapes and attributes.

```python
yellow = RGB(233, 196, 106)
charcoal = RGB(38, 70, 83)
circle(1).set_fill_color(yellow).set_stroke_color(charcoal)
```
![circle](https://github.com/danoneata/pydiagrams/blob/master/examples/circle.png)

Combining diagrams.

```python
aqua = RGB(0, 255, 255)
circle(1) + square(1).set_fill_color(aqua)
```
![atop](https://github.com/danoneata/pydiagrams/blob/master/examples/atop.png)

Combining diagrams, horizontal composition:

```python
Diagram.hcat(circle(0.1 * i) for i in range(1, 6))
```
![hcat](https://github.com/danoneata/pydiagrams/blob/master/examples/hcat.png)

## More examples

```python
streamlit run examples/squares.py
```

![squares](https://github.com/danoneata/pydiagrams/blob/master/examples/squares.png)

```python
streamlit run examples/escher_square_limit.py
```

![escher](https://github.com/danoneata/pydiagrams/blob/master/examples/escher_square_limit.png)

## TODO

- [ ] Finish refactoring
    - [ ] Transform bounding boxes
- [ ] Using bounding boxes as extents is not composable (rotate 45); for the moment restrict rotations to 90?
- [ ] Composing transformations is not efficient.
- [ ] Draw horizontal and vertical lines
- [ ] Rename blank to phantom?
- [ ] Allow change of file type for the `render` function
- [ ] Add `rotate_by` transformation (rotates by a fraction of the circle)
- [ ] Define color names
- [ ] Allow for HSV colors
- [ ] Add more examples: e.g., Vera Molnár's drawings, Hilbert curve, Escher's Square Limit
- [ ] How to allow the context?
- [x] Add `Scale` transformation
- [x] Show origin helper function
- [x] Update width and height of image to fit all the contents
