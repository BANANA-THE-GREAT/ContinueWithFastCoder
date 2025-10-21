"""
    twtxt.parser
    ~~~~~~~~~~~~

    This module implements the parser for twtxt files.

    :copyright: (c) 2016-2022 by buckket.
    :license: MIT, see LICENSE for more details.
"""

import logging
from datetime import datetime, timezone

import click
import dateutil.parser



logger = logging.getLogger(__name__)


def make_aware(dt):
    """Appends tzinfo and assumes UTC, if datetime object has no tzinfo already."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def parse_iso8601(string):
    """Parse string using dateutil.parser."""
    return make_aware(dateutil.parser.parse(string))





























def parse_tweet(raw_tweet, source, now=None):
    """
        Parses a single raw tweet line from a twtxt file
        and returns a :class:`Tweet` object.

        :param str raw_tweet: a single raw tweet line
        :param Source source: the source of the given tweet
        :param Datetime now: the current datetime

        :returns: the parsed tweet
        :rtype: Tweet
    """
    from twtxt.models import Tweet
    if now is None:
        now = datetime.now(timezone.utc)

    raw_created_at, text = raw_tweet.split("\t", 1)
    created_at = parse_iso8601(raw_created_at)

    if created_at > now:
        raise ValueError("Tweet is from the future")

    return Tweet(click.unstyle(text.strip()), created_at, source)
