"""I/O utilities for benchmark scripts"""

import os
import sys
import contextlib


@contextlib.contextmanager
def suppress_c_stdout_stderr():
    """A context manager that redirects C-level stdout and stderr to devnull."""
    # File descriptor for stdout and stderr
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()

    # Save original file descriptors
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)

    # Open devnull
    devnull_fd = os.open(os.devnull, os.O_RDWR)

    try:
        # Redirect stdout and stderr to devnull
        os.dup2(devnull_fd, stdout_fd)
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        # Restore stdout and stderr
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        # Close saved file descriptors
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


def format_number(num):
    """Format number with thousands separators"""
    return f"{int(num):,}"

