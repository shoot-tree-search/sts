"""Operating system utilities."""

import contextlib
import shutil
import tempfile


@contextlib.contextmanager
def atomic_dump(paths):
    """Enables dumping data to several files atomically.

    Useful for dumping big files. If the write is interrupted, the file is not
    left in an invalid state. This is done by first writing to a temporary file
    and then substituting it for the target path.

    Args:
        paths (tuple): Tuple of target paths.

    Yields:
        Tuple of paths to the temporary file to write to.
    """
    with contextlib.ExitStack() as stack:
        # Create the temporary files. They will all be closed at the end of the
        # `with` block.
        tmp_paths = tuple(
            stack.enter_context(tempfile.NamedTemporaryFile()).name  # pylint: disable=no-member
            for _ in paths
        )
        # Write to the temporary files.
        yield tmp_paths
        # Overwrite the old files with the new ones.
        for (path, tmp_path) in zip(paths, tmp_paths):
            shutil.move(tmp_path, path)
            # Create an empty file to avoid a FileNotFoundError on exit.
            open(tmp_path, 'w').close()
