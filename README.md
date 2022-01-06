An exploration into a declarative API for drawing over PyCairo.
The design draws inspiration from the [diagrams](https://diagrams.github.io/) library in Haskell.

## Examples

```python
streamlit run examples/squares.py
```

![squares](https://github.com/danoneata/pydiagrams/examples/squares.png)

```python
streamlit run examples/escher_square_limit.py
```

## TODO

- [ ] Update width and height of image to fit all the contents
- [ ] Do not use absolute coördinates
- [ ] Show origin helper function
- [ ] Allow change of backend for the `render` function
- [ ] Add `Scale` transformation
- [ ] Add `RotateBy` transformation (rotates by a fraction of the circle)
- [ ] Define color names
- [ ] Allow for HSV colors
- [ ] Add more examples: e.g., Vera Molnár's drawings, Hilbert curve, Escher's Square Limit
