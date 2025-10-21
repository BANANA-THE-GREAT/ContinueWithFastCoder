# Copyright (c) 2015, Menno Smits
# Released subject to the New BSD License
# Please see http://en.wikipedia.org/wiki/BSD_licenses

import logging
from typing import Iterator, Optional, Tuple, Union



logger = logging.getLogger(__name__)


def to_unicode(s: Union[bytes, str]) -> str:
    if isinstance(s, bytes):
        try:
            return s.decode("ascii")
        except UnicodeDecodeError:
            logger.warning(
                "An error occurred while decoding %s in ASCII 'strict' mode. Fallback to "
                "'ignore' errors handling, some characters might have been stripped",
                s,
            )
            return s.decode("ascii", "ignore")
    return s


def to_bytes(s: Union[bytes, str], charset: str = "ascii") -> bytes:
    if isinstance(s, str):
        return s.encode(charset)
    return s














_TupleAtomPart = Union[None, int, bytes]
_TupleAtom = Tuple[Union[_TupleAtomPart, "_TupleAtom"], ...]


def chunk(lst: _TupleAtom, size: int) -> Iterator[_TupleAtom]:
    for i in range(0, len(lst), size):
        yield lst[i : i + size]
