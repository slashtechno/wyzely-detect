import argparse
import os
import dotenv
from pathlib import Path

argparser = None


def set_argparse():
    global argparser

    if Path(".env").is_file():
        dotenv.load_dotenv()
        print("Loaded .env file")
    else:
        print("No .env file found")


    # One important thing to consider is that most function parameters are optional and have a default value
    # However, with argparse, those are never used since a argparse always passes something, even if it's None
    argparser = argparse.ArgumentParser(
        prog="Wyzely Detect",
        description="Recognize faces/objects in a video stream (from a webcam or a security camera) and send notifications to your devices",  # noqa: E501
        epilog=":)",
    )


    video_options = argparser.add_argument_group("Video Options")
    stream_source = video_options.add_mutually_exclusive_group()
    stream_source.add_argument(
        "--rtsp-url",
        default=os.environ["RTSP_URL"]
        if "RTSP_URL" in os.environ and os.environ["RTSP_URL"] != ""
        else None,  # noqa: E501
        type=str,
        help="RTSP camera URL",
    )
    stream_source.add_argument(
        "--capture-device",
        default=os.environ["CAPTURE_DEVICE"]
        if "CAPTURE_DEVICE" in os.environ and os.environ["CAPTURE_DEVICE"] != ""
        else 0,  # noqa: E501
        type=int,
        help="Capture device number",
    )
    video_options.add_argument(
        "--run-scale",
        # Set it to the env RUN_SCALE if it isn't blank, otherwise set it to 0.25
        default=os.environ["RUN_SCALE"]
        if "RUN_SCALE" in os.environ and os.environ["RUN_SCALE"] != ""
        # else 0.25,
        else 1,
        type=float,
        help="The scale to run the detection at, default is 0.25",
    )
    video_options.add_argument(
        "--view-scale",
        # Set it to the env VIEW_SCALE if it isn't blank, otherwise set it to 0.75
        default=os.environ["VIEW_SCALE"]
        if "VIEW_SCALE" in os.environ and os.environ["VIEW_SCALE"] != ""
        # else 0.75,
        else 1,
        type=float,
        help="The scale to view the detection at, default is 0.75",
    )

    video_options.add_argument(
        "--no-display",
        default=os.environ["NO_DISPLAY"]
        if "NO_DISPLAY" in os.environ and os.environ["NO_DISPLAY"] != ""
        else False,
        action="store_true",
        help="Don't display the video feed",
    )


    notifcation_services = argparser.add_argument_group("Notification Services")
    notifcation_services.add_argument(
        "--ntfy-url",
        default=os.environ["NTFY_URL"]
        if "NTFY_URL" in os.environ and os.environ["NTFY_URL"] != ""
        else None,
        type=str,
        help="The URL to send notifications to",
    )

    timers = argparser.add_argument_group("Timers")
    timers.add_argument(
        "--detection-duration",
        default=os.environ["DETECTION_DURATION"]
        if "DETECTION_DURATION" in os.environ and os.environ["DETECTION_DURATION"] != ""
        else 2,
        type=int,
        help="The duration (in seconds) that an object must be detected for before sending a notification",
    )
    timers.add_argument(
        "--detection-window",
        default=os.environ["DETECTION_WINDOW"]
        if "DETECTION_WINDOW" in os.environ and os.environ["DETECTION_WINDOW"] != ""
        else 15,
        type=int,
        help="The time (seconds) before the detection duration resets",
    )
    timers.add_argument(
        "--notification-window",
        default=os.environ["NOTIFICATION_WINDOW"]
        if "NOTIFICATION_WINDOW" in os.environ
        and os.environ["NOTIFICATION_WINDOW"] != ""
        else 30,
        type=int,
        help="The time (seconds) before another notification can be sent",
    )


    face_recognition = argparser.add_argument_group("Face Recognition options")
    face_recognition.add_argument(
        "--faces-directory",
        default=os.environ["FACES_DIRECTORY"]
        if "FACES_DIRECTORY" in os.environ and os.environ["FACES_DIRECTORY"] != ""
        else "faces",
        type=str,
        help="The directory to store the faces. Can either contain images or subdirectories with images, the latter being the preferred method",  # noqa: E501
    )
    face_recognition.add_argument(
        "--face-confidence-threshold",
        default=os.environ["FACE_CONFIDENCE_THRESHOLD"]
        if "FACE_CONFIDENCE_THRESHOLD" in os.environ
        and os.environ["FACE_CONFIDENCE_THRESHOLD"] != ""
        else 0.3,
        type=float,
        help="The confidence (currently cosine similarity) threshold to use for face recognition",
    )
    face_recognition.add_argument(
        "--no-remove-representations",
        default=os.environ["NO_REMOVE_REPRESENTATIONS"]
        if "NO_REMOVE_REPRESENTATIONS" in os.environ
        and os.environ["NO_REMOVE_REPRESENTATIONS"] != ""
        else False,
        action="store_true",
        help="Don't remove representations_<model>.pkl at the start of the program. Greatly improves startup time, but doesn't take into account changes to the faces directory since it was created",  # noqa: E501
    )



    object_detection = argparser.add_argument_group("Object Detection options")
    object_detection.add_argument(
        "--detect-object",
        nargs="*",
        default=[],
        type=str,
        help="The object(s) to detect. Must be something the model is trained to detect",
    )
    object_detection.add_argument(
        "--object-confidence-threshold",
        default=os.environ["OBJECT_CONFIDENCE_THRESHOLD"]
        if "OBJECT_CONFIDENCE_THRESHOLD" in os.environ
        and os.environ["OBJECT_CONFIDENCE_THRESHOLD"] != ""
        else 0.6,
        type=float,
        help="The confidence threshold to use",
    )

    # return argparser


# This will run when this file is imported
set_argparse()
