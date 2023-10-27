import httpx
import time


"""
Structure of objects_and_peoples
Really, the only reason peoples is a separate dictionary is to prevent duplicates, though it just makes the code more complicated.
{
    "objects": {
        "object_name": {
            "last_detection_time": float,
            "detection_duration": float,
            "last_notification_time": float,
            },
        },
    "peoples": {
        "person_name": {
            "last_detection_time": float,
            "detection_duration": float,
            "last_notification_time": float,
            },
        },
}
"""
# objects_and_peoples = {}


def thing_detected(
    thing_name: str,
    objects_and_peoples: dict,
    detection_type: str = "objects",
    detection_window: int = 15,
    detection_duration: int = 2,
    notification_window: int = 15,
    ntfy_url: str = "https://ntfy.sh/wyzely-detect",
) -> dict:
    """
    A function to make sure 2 seconds of detection is detected in 15 seconds, 15 seconds apart.
    Takes a dict that will be retured with the updated detection times. MAKE SURE TO SAVE THE RETURNED DICTIONARY
    """

    # "Alias" the objects and peoples dictionaries so it's easier to work with
    respective_type = objects_and_peoples[detection_type]

    # (re)start cycle
    try:
        if (
            # If the object has not been detected before
            respective_type[thing_name]["last_detection_time"] is None
            # If the last detection was more than 15 seconds ago
            or time.time() - respective_type[thing_name]["last_detection_time"]
            > detection_window
        ):
            # Set the last detection time to now
            respective_type[thing_name]["last_detection_time"] = time.time()
            print(f"First detection of {thing_name} in this detection window")
            # This line is important. It resets the detection duration when the object hasn't been detected for a while
            # If detection duration is None, don't print anything.
            # Otherwise, print that the detection duration is being reset due to inactivity
            if respective_type[thing_name]["detection_duration"] is not None:
                print(
                    f"Resetting detection duration for {thing_name} since it hasn't been detected for {detection_window} seconds"  # noqa: E501
                )
            respective_type[thing_name]["detection_duration"] = 0
        else:
            # Check if the last NOTIFICATION was less than 15 seconds ago
            # If it was, then don't do anything
            if (
                time.time() - respective_type[thing_name]["last_detection_time"]
                <= notification_window
            ):
                pass
            # If it was more than 15 seconds ago, reset the detection duration
            # This effectively resets the notification timer
            else:
                print("Notification timer has expired - resetting")
                respective_type[thing_name]["detection_duration"] = 0
            respective_type[thing_name]["detection_duration"] += (
                time.time() - respective_type[thing_name]["last_detection_time"]
            )
            # print("Updating detection duration")
            respective_type[thing_name]["last_detection_time"] = time.time()
    except KeyError:
        # If the object has not been detected before
        respective_type[thing_name] = {
            "last_detection_time": time.time(),
            "detection_duration": 0,
            "last_notification_time": None,
        }
        print(f"First detection of {thing_name} ever")

    # (re)send notification
    # Check if detection has been ongoing for 2 seconds or more in the past 15 seconds
    if (
        respective_type[thing_name]["detection_duration"] >= detection_duration
        and time.time() - respective_type[thing_name]["last_detection_time"]
        <= detection_window
    ):
        # If the last notification was more than 15 seconds ago, then send a notification
        if (
            respective_type[thing_name]["last_notification_time"] is None
            or time.time() - respective_type[thing_name]["last_notification_time"]
            > notification_window
        ):
            respective_type[thing_name]["last_notification_time"] = time.time()
            print(f"Detected {thing_name} for {detection_duration} seconds")
            if ntfy_url is None:
                print(
                    "ntfy_url is None. Not sending notification. Set ntfy_url to send notifications"
                )
            else:
                headers = construct_ntfy_headers(
                    title=f"{thing_name} detected",
                    tag="rotating_light",
                    priority="default",
                )
                send_notification(
                    data=f"{thing_name} detected for {detection_duration} seconds",
                    headers=headers,
                    url=ntfy_url,
                )
                # Reset the detection duration
                print("Just sent a notification - resetting detection duration")
            respective_type[thing_name]["detection_duration"] = 0

        # Take the aliased objects_and_peoples and update the respective dictionary
        objects_and_peoples[detection_type] = respective_type
    return objects_and_peoples


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
