# Copyright (c) 2014, Menno Smits
# Released subject to the New BSD License
# Please see http://en.wikipedia.org/wiki/BSD_licenses

import re
from datetime import datetime
from email.utils import parsedate_tz

from .fixed_offset import FixedOffset

_SHORT_MONTHS = " Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(" ")



























def datetime_to_native(dt: datetime) -> datetime:
    return dt.astimezone(FixedOffset.for_system()).replace(tzinfo=None)














# Matches timestamp strings where the time separator is a dot (see
# issue #154). For example: 'Sat, 8 May 2010 16.03.09 +0200'
_rfc822_dotted_time = re.compile(r"\w+, ?\d{1,2} \w+ \d\d(\d\d)? \d\d?\.\d\d?\.\d\d?.*")


def _munge(timestamp: bytes) -> str:
    s = timestamp.decode("latin-1")  # parsedate_tz only works with strings
    if _rfc822_dotted_time.match(s):
        return s.replace(".", ":")
    return s






