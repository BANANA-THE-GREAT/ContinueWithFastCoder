import urllib.parse

from zulipterminal.api_types import Message


def hash_util_encode(string: str) -> str:
    """
    Hide URI-encoding by replacing '%' with '.'
    urllib.quote is equivalent to encodeURIComponent in JavaScript.
    Referred from zerver/lib/url_encoding.py
    """
    # `safe` has a default value of "/", but we want those encoded, too.
    return urllib.parse.quote(string, safe=b"").replace(".", "%2E").replace("%", ".")












def near_stream_message_url(server_url: str, message: Message) -> str:
    """
    Returns the complete encoded URL of a message from #narrow/stream.
    Referred from zerver/lib/url_encoding.py.
    """
    message_id = str(message["id"])
    stream_id = message["stream_id"]
    stream_name = message["display_recipient"]
    topic_name = message["subject"]
    encoded_stream = encode_stream(stream_id, stream_name)
    encoded_topic = hash_util_encode(topic_name)

    parts = [
        server_url,
        "#narrow",
        "stream",
        encoded_stream,
        "topic",
        encoded_topic,
        "near",
        message_id,
    ]
    full_url = "/".join(parts)
    return full_url


def near_pm_message_url(server_url: str, message: Message) -> str:
    """
    Returns the complete encoded URL of a message from #narrow/pm-with.
    Referred from zerver/lib/url_encoding.py.
    """
    message_id = str(message["id"])
    str_user_ids = [str(recipient["id"]) for recipient in message["display_recipient"]]

    pm_str = ",".join(str_user_ids) + "-pm"
    parts = [
        server_url,
        "#narrow",
        "pm-with",
        pm_str,
        "near",
        message_id,
    ]
    full_url = "/".join(parts)
    return full_url













