import os
import sys
import tempfile
import time
from typing import NoReturn, Union, Optional, Any, TypeVar
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


def show(filepath: str):  # type: ignore
    """Show image from filepath.

    Args:
        filepath (str): Filepath of the image.
                        example: "examples/output/intro-01-a.png"
    """
    PILImage.open(filepath).show()


def imgen(
    d: "Diagram",
    temporary: bool = True,
    dirpath: Optional[str] = "examples/output",
    prefix: str = "trial_",
    suffix: str = "_image.png",
    height: int = 64,
    wait: int = 5,
    verbose: bool = True,
) -> Union[NoReturn, None]:
    """Render a ``chalk`` diagram and visualize.

    Args:
        d (Diagram): A chalk diagram object (``chalk.Diagram``).
        temporary (bool, optional): Whether to use a temporary file or not.
                Defaults to True.
        dirpath (Optional[str], optional): Directory to save the temporary
                file in. Defaults to "examples/output".
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

    Returns:
        Union[NoReturn, None]: Does not return anything.

    Usage:
        from colour import Color
        from chalk import circle

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
    """
    make_tempdir = False
    dp = None
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
                    f" ✅ 1. Created temporary file: {os.path.relpath(fp.name)}"
                )  # noqa: E501
            d.render(fp.name, height=height)
            if verbose:
                prnt_success(" ✅ 2. Saved rendered image to temporary file.")
            fp.seek(0)
            if verbose:
                prnt_success(" ✅ 3. Displaying image from temporary file.")
            show(fp.name)
            time.sleep(wait)

        if verbose:
            prnt_success(" ✅ 4. Closed and removed temporary image!")

        if make_tempdir and dp:
            # Cleanup temporary directory
            dp.cleanup()
    else:
        raise NotImplementedError(
            "Only temporary file creation + load + display is supported."
        )


def create_sample_diagram() -> "Diagram":
    """Creates a sample diagram.

    Returns:
        Diagram: Returns a sample diagram.
    """
    from colour import Color
    from chalk import circle, square

    papaya = Color("#ff9700")
    blue = Color("#005FDB")
    d = circle(0.5).fill_color(papaya).beside(square(1).fill_color(blue))

    return d


def quick_probe(
    d: Optional["Diagram"] = None,
    dirpath: Optional[str] = None,
    verbose: bool = True,
    **kwargs: Any,
) -> Union[NoReturn, None]:
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
        verbose (bool): Set verbosity. Defaults to True.

    Usage:
        quick_probe(verbose=True)
    """
    if verbose:
        prnt_warning(f"{chalk.__name__} version: v{chalk.__version__}")
    if d is None:
        d = create_sample_diagram()
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
