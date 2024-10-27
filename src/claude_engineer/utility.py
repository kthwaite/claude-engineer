"""Utility functions for Claude Engineer."""

import shutil
import os
from typing import Any, AsyncIterable, TypeVar


def is_installed(lib_name):
    return shutil.which(lib_name) is not None


async def text_chunker(text: str) -> AsyncIterable[str]:
    """Split text into chunks, ensuring to not break sentences."""
    splitters = (".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " ")
    buffer = ""

    for char in text:
        if buffer.endswith(splitters):
            yield buffer + " "
            buffer = char
        elif char in splitters:
            yield buffer + char + " "
            buffer = ""
        else:
            buffer += char

    if buffer:
        yield buffer + " "


def get_env_checked(key, default: str | None = None) -> str:
    """Get an environment variable, raising an error if not found."""
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"{key} not found in environment variables")
    return value
