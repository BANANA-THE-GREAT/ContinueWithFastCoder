# Copyright (c) 2014, Menno Smits
# Released subject to the New BSD License
# Please see http://en.wikipedia.org/wiki/BSD_licenses

import datetime
import time
from typing import Optional

ZERO = datetime.timedelta(0)


class FixedOffset(datetime.tzinfo):
    """
    This class describes fixed timezone offsets in hours and minutes
    east from UTC
    """

    def __init__(self, minutes: float) -> None:
        self.__offset = datetime.timedelta(minutes=minutes)

        sign = "+"
        if minutes < 0:
            sign = "-"
        hours, remaining_mins = divmod(abs(minutes), 60)
        self.__name = "%s%02d%02d" % (sign, hours, remaining_mins)

    def utcoffset(self, _: Optional[datetime.datetime]) -> datetime.timedelta:
        return self.__offset

    def tzname(self, _: Optional[datetime.datetime]) -> str:
        return self.__name

    def dst(self, _: Optional[datetime.datetime]) -> datetime.timedelta:
        return ZERO

    @classmethod









