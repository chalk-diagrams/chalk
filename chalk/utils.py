"""
The ``chalk.utils`` module is meant to provide various
utility-oriented functionalities.

Importing:

    ```python
    # method-1
    from chalk import utils as U

    # method-2
    import chalk.utils

    # method-3
    from chalk.utils import <some_function>

    ```
"""

import os
import sys
import tempfile
import time
from typing import Any, Optional, Tuple, TypeVar, Union

from colour import Color
from PIL import Image as PILImage

import chalk

_HERE = os.path.dirname(__file__)

try:
    from loguru import logger

    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="<light-red>{time:HH:mm:ss}</light-red> <level>{message}</level>",  # noqa: E501
        level="INFO",
    )  # noqa: E124
    prnt_success = logger.success
    prnt_warning = logger.warning
except ImportError:
    prnt_success = print  # type: ignore
    prnt_warning = print  # type: ignore

Diagram = TypeVar("Diagram")


def show(filepath: str) -> None:
    """Show image from filepath.

    Args:
        filepath (str): Filepath of the image.
                        example: "examples/output/intro-01-a.png"

    Usage:

        ```python
        from chalk.utils import show
        show("examples/output/intro-01.png")
        ```
    """
    PILImage.open(filepath).show()


def imgen(
    d: Diagram,
    temporary: bool = True,
    dirpath: Optional[str] = "examples/output",
    prefix: str = "trial_",
    suffix: str = "_image.png",
    height: int = 64,
    wait: int = 5,
    verbose: bool = True,
) -> None:
    """Render a ``chalk`` diagram and visualize.

    Args:
        d (Diagram): A chalk diagram object (``chalk.Diagram``).
        temporary (bool, optional): Whether to use a temporary file or not.
                Defaults to True.
        dirpath (Optional[str], optional): Directory to save the temporary
                file in. If does not exist, creates a temporary directory
                and destroys it afterwards. Defaults to "examples/output".
        prefix (str, optional): Prefix for the generated image file.
                Defaults to "trial_".
        suffix (str, optional): Suffix for the generated image file.
                Defaults to "_image.png".
        height (int, optional): Height of the diagram, rendered as an image.
                Defaults to 64.
        wait (int, optional): The time (in seconds) to wait until destroying
                the temporary image file. Defaults to 5.
        verbose (bool): Set verbosity. Defaults to True.

    Raises:
        NotImplementedError: For non temporary file (``temporary=False``),
                             raises an error, as it has not been
                             implemented yet.

    Usage:

        ```python
        from colour import Color
        from chalk import circle
        from chalk.utils import imgen

        papaya = Color("#ff9700")
        d = circle(0.5).fill_color(papaya)

        # Minimal example
        imgen(d, temporary=True)

        # Temporary file is created in current directory
        imgen(d, temporary=True, dirpath=None)

        # Folder path must exist; otherwise temporary folder is used
        imgen(d, temporary=True, dirpath="examples/output")

        # Display and delete the temporary file after 10 seconds
        imgen(d, temporary=True, wait=10)
        ```
    """
    make_tempdir = False
    dp = None
    if verbose:
        prnt_warning(f" ✨ {chalk.__name__} version: v{chalk.__version__}")
    if temporary:
        if (dirpath is not None) and (not os.path.isdir(dirpath)):
            make_tempdir = True
            dp = tempfile.TemporaryDirectory(
                dir=".", prefix=prefix, suffix=suffix
            )
            dirpath = dp.name
        with tempfile.NamedTemporaryFile(
            dir=dirpath, prefix=prefix, suffix=suffix
        ) as fp:
            if verbose:
                prnt_success(
                    f" ✅ 1. Created temporary file: \n\t\t{os.path.relpath(fp.name)}"  # noqa: E501
                )  # noqa: E501
            d.render(fp.name, height=height)  # type: ignore
            if verbose:
                prnt_success(" ✅ 2. Saved rendered image to temporary file.")
            fp.seek(0)
            if verbose:
                prnt_success(" ✅ 3. Displaying image from temporary file.")
            show(fp.name)
            time.sleep(wait)

        if verbose:
            prnt_success(" ✅ 4. Removed temporary image file!")

        if make_tempdir and dp:
            # Cleanup temporary directory
            dp.cleanup()
    else:
        raise NotImplementedError(
            "Only temporary file creation + load + display is supported."
        )


def create_sample_diagram(
    option: Optional[str] = "a|b",
) -> Union[Diagram, Tuple[Diagram, Diagram]]:
    """Creates a sample diagram.

    Args:
        option (Optional[str], optional): A string denoting what
            kind of sample diagram(s) to return.
            💡 Defaults to ``"a|b"``.

    Choose ``option`` from for the following.

    Click to expand:

        |   Option    |      Meaning      |     Output      |
        |:-----------:|:------------------|:---------------:|
        | ``"a+b"``   | ``a.atop(b)``     | Single Diagram  |
        | ``"b+a"``   | ``b.atop(a)``     | Single Diagram  |
        | ``"a|b"``   | ``a.beside(b)``   | Single Diagram  |
        | ``"b|a"``   | ``b.beside(a)``   | Single Diagram  |
        | ``"a/b"``   | ``a.above(b)``    | Single Diagram  |
        | ``"b/a"``   | ``b.above(a)``    | Single Diagram  |
        | ``"a//b"``  | ``a.above2(b)``   | Single Diagram  |
        | ``"b//a"``  | ``b.above2(a)``   | Single Diagram  |
        | ``"a,b"``   | ``(a, b)``        | Two Diagrams    |

    Returns:
        Diagram: Returns a sample diagram.

    Usage:

        ```python
        from chalk.utils import create_sample_diagram

        # create a diagram composed of two diagrams: a|b
        d = create_sample_diagram(option="a|b")

        # create a diagram composed of two diagrams: b|a
        d = create_sample_diagram(option="b|a")

        # create a diagram composed of two diagrams: a+b
        d = create_sample_diagram(option="a+b")

        # create a diagram composed of two diagrams: a/b
        d = create_sample_diagram(option="a/b")

        # create a diagram composed of two diagrams: a//b
        d = create_sample_diagram(option="a//b")

        # create two diagrams: (a,b)
        a, b = create_sample_diagram(option="a,b")
        ```
    """
    from chalk import circle, square

    papaya = Color("#ff9700")
    blue = Color("#005FDB")
    a = circle(0.5).fill_color(papaya)
    b = square(1).fill_color(blue)

    if option is None:
        d = a | b  # a|b
    else:
        option = "".join(option.split())
        # handle specific cases
        if option == "a+b":
            d = a + b  # a.atop(b)
        if option == "b+a":
            d = b + a  # b.atop(a)
        elif option == "a|b":
            d = a | b  # a.beside(b)
        elif option == "a|b":
            d = b | a  # b.beside(a)
        elif option == "a/b":
            d = a / b  # a.above(b)
        elif option == "b/a":
            d = b / a  # b.above(a)
        elif option == "a//b":
            d = a // b  # a.above2(b)
        elif option == "b//a":
            d = b // a  # b.above2(a)
        elif option == "a,b":
            d = (a, b)  # type: ignore
        elif option == "b,a":
            d = (b, a)  # type: ignore
    return d  # type: ignore


def create_double_diagrams() -> Tuple[Diagram, Diagram]:
    """Creates a pair of sample diagrams (a circle and a square).

    Returns:
        Diagram: Returns a sample diagram.
    """
    a, b = create_sample_diagram(option="a,b")  # type: ignore

    return (a, b)


def quick_probe(
    d: Optional[Diagram] = None,
    dirpath: Optional[str] = None,
    verbose: bool = True,
    **kwargs: Any,
) -> None:
    """Render diagram and generate an image tempfile (``.png``)

    This utility is made to quickly create a sample diagram and display it,
    without saving any permanent image file on disk. If a diagram is not
    provided, a sample diagram is generated. If a diagram is provided, it
    is displayed.

    Args:
        d (Optional[Diagram], optional): A chalk diagram object
                (``chalk.Diagram``). Defaults to None.
        dirpath (Optional[str], optional): Directory to save the temporary
                file in. For example, you could use "examples/output" with
                respect to the location of running a script.
                Defaults to None.
        verbose (bool, optional): Set verbosity. Defaults to True.
        **kwargs (Any, optional): See the keyword arguments of
                                  [``imgen()``][chalk.utils.imgen].

    Usage:

        ```python
        from chalk.utils import quick_probe
        quick_probe(verbose=True, wait=2)
        ```
    """
    # if verbose:
    #     prnt_warning(f"{chalk.__name__} version: v{chalk.__version__}")
    if d is None:
        d = create_sample_diagram()  # type: ignore
    if dirpath is None:
        dirpath = os.path.join(_HERE, "../examples/output")
    # render diagram and generate an image tempfile (.png)
    imgen(d, dirpath=dirpath, verbose=verbose, **kwargs)


if __name__ == "__main__":

    # determine initial directory
    root = os.path.abspath(os.curdir)
    # update sys-path
    sys.path.append(_HERE)
    os.chdir(_HERE)  # change directory
    quick_probe(verbose=True)  # generate diagram
    os.chdir(root)  # switch back to initial directory
