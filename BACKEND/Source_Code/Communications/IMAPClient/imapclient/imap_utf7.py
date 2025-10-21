# This file contains two main methods used to encode and decode UTF-7
# string, described in the RFC 3501. There are some variations specific
# to IMAP4rev1, so the built-in Python UTF-7 codec can't be used instead.
#
# The main difference is the shift character (used to switch from ASCII to
# base64 encoding context), which is & in this modified UTF-7 convention,
# since + is considered as mainly used in mailbox names.
# Other variations and examples can be found in the RFC 3501, section 5.1.3.

import binascii
from typing import List, Union














































AMPERSAND_ORD = ord("&")
DASH_ORD = ord("-")









































def base64_utf7_encode(buffer: List[str]) -> bytes:
    s = "".join(buffer).encode("utf-16be")
    return binascii.b2a_base64(s).rstrip(b"\n=").replace(b"/", b",")


def base64_utf7_decode(s: bytearray) -> str:
    s_utf7 = b"+" + s.replace(b",", b"/") + b"-"
    return s_utf7.decode("utf-7")
