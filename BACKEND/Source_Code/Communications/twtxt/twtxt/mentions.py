"""
    twtxt.mentions
    ~~~~~~~~~~~~~~

    This module implements functions for handling mentions in twtxt.

    :copyright: (c) 2016-2022 by buckket.
    :license: MIT, see LICENSE for more details.
"""

import re

import click

mention_re = re.compile(r'@<(?:(?P<name>\S+?)\s+)?(?P<url>\S+?://.*?)>')
short_mention_re = re.compile(r'@(?P<name>\w+)')


def get_source_by_url(url):
    conf = click.get_current_context().obj["conf"]
    if url == conf.twturl:
        return conf.source
    return next((source for source in conf.following if url == source.url), None)


def get_source_by_name(nick):
    nick = nick.lower()
    conf = click.get_current_context().obj["conf"]
    if nick == conf.nick and conf.twturl:
        return conf.source
    return next((source for source in conf.following if nick == source.nick), None)


def expand_mentions(text, embed_names=True):
    """Searches the given text for mentions and expands them.

    For example:
    "@source.nick" will be expanded to "@<source.nick source.url>".
    """
    if embed_names:
        mention_format = "@<{name} {url}>"
    else:
        mention_format = "@<{url}>"

    def handle_mention(match):
        source = get_source_by_name(match.group(1))
        if source is None:
            return "@{0}".format(match.group(1))
        return mention_format.format(
            name=source.nick,
            url=source.url)

    return short_mention_re.sub(handle_mention, text)


def format_mention(name, url):
    source = get_source_by_url(url)
    if source:
        if source.nick == click.get_current_context().obj["conf"].nick:
            return click.style("@{0}".format(source.nick), fg="magenta", bold=True)
        else:
            return click.style("@{0}".format(source.nick), bold=True)
    elif name:
        return "@{0}".format(name)
    else:
        return "@<{0}>".format(url)



















