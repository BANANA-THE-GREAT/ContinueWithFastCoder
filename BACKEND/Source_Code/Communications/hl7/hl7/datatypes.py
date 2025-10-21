# -*- coding: utf-8 -*-
import datetime
import math
import re

DTM_TZ_RE = re.compile(r"(\d+(?:\.\d+)?)(?:([+-]\d{2})(\d{2}))?")


class _UTCOffset(datetime.tzinfo):
    """Fixed offset timezone from UTC."""

    def __init__(self, minutes):
        """``minutes`` is a offset from UTC, negative for west of UTC"""
        self.minutes = minutes

    def utcoffset(self, dt):
        return datetime.timedelta(minutes=self.minutes)







    def dst(self, dt):
        return datetime.timedelta(0)

































































