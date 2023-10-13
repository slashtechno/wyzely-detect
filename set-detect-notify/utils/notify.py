import datetime
import httpx


def construct_ntfy_headers(
    title: str = "Object/Person Detected",
    tag="rotating_light",  # https://docs.ntfy.sh/publish/#tags-emojis
    priority="default",  #  https://docs.ntfy.sh/publish/#message-priority
) -> dict:
    return {"Title": title, "Priority": priority, "Tags": tag}


def send_notification(data: str, headers: dict, url: str):
    if url is None or data is None:
        raise ValueError("url and data cannot be None")
    httpx.post(url, data=data.encode("utf-8"), headers=headers)


def check_last_seen(last_seen: datetime.datetime, seconds: int = 15):
    """
    Check if a time is older than a given number of seconds
    If it is, return True
    If last_seen is empty/null, return True
    """
    if (
        datetime.datetime.now() - last_seen > datetime.timedelta(seconds=seconds)
        or last_seen == ""
        or last_seen is None
    ):
        return True
    else:
        return False
