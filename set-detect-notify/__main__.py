# import face_recognition
import cv2
import numpy as np
import dotenv
from pathlib import Path
import os
import time

# import hjson as json
import torch
from ultralytics import YOLO

import argparse

from .utils import notify
from .utils import utils

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
args = None

object_names = {}


def main():
    global object_names
    global args
    # RUN_BY_COMPOSE = os.getenv("RUN_BY_COMPOSE") # Replace this with code to check for gpu

    if Path(".env").is_file():
        dotenv.load_dotenv()
        print("Loaded .env file")
    else:
        print("No .env file found")

    argparser = argparse.ArgumentParser(
        prog="Detect It",
        description="Detect it all!",
        epilog=":)",
    )

    # required='RUN_SCALE' not in os.environ,

    argparser.add_argument(
        "--run-scale",
        # Set it to the env RUN_SCALE if it isn't blank, otherwise set it to 0.25
        default=os.environ["RUN_SCALE"]
        if "RUN_SCALE" in os.environ and os.environ["RUN_SCALE"] != ""
        else 0.25,  # noqa: E501
        type=float,
        help="The scale to run the detection at, default is 0.25",
    )
    argparser.add_argument(
        "--view-scale",
        # Set it to the env VIEW_SCALE if it isn't blank, otherwise set it to 0.75
        default=os.environ["VIEW_SCALE"]
        if "VIEW_SCALE" in os.environ and os.environ["VIEW_SCALE"] != ""
        else 0.75,  # noqa: E501
        type=float,
        help="The scale to view the detection at, default is 0.75",
    )

    argparser.add_argument(
        "--confidence-threshold",
        default=os.environ["CONFIDENCE_THRESHOLD"]
        if "CONFIDENCE_THRESHOLD" in os.environ
        and os.environ["CONFIDENCE_THRESHOLD"] != ""
        else 0.6,
        type=float,
        help="The confidence threshold to use",
    )

    argparser.add_argument(
        "--detect-object",
        nargs="*",
        default=[],
        type=str,
        help="The object(s) to detect. Must be something the model is trained to detect",
    )

    stream_source = argparser.add_mutually_exclusive_group()
    stream_source.add_argument(
        "--url",
        default=os.environ["URL"]
        if "URL" in os.environ and os.environ["URL"] != ""
        else None,  # noqa: E501
        type=str,
        help="The URL of the stream to use",
    )
    stream_source.add_argument(
        "--capture-device",
        default=os.environ["CAPTURE_DEVICE"]
        if "CAPTURE_DEVICE" in os.environ and os.environ["CAPTURE_DEVICE"] != ""
        else 0,  # noqa: E501
        type=int,
        help="The capture device to use. Can also be a url.",
    )

    notifcation_services = argparser.add_argument_group("Notification Services")
    notifcation_services.add_argument(
        "--ntfy-url",
        default=os.environ["NTFY_URL"]
        if "NTFY_URL" in os.environ and os.environ["NTFY_URL"] != ""
        else "https://ntfy.sh/set-detect-notify",
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

    args = argparser.parse_args()

    # Check if a CUDA GPU is available. If it is, set it via torch. Ff not, set it to cpu
    # https://github.com/ultralytics/ultralytics/issues/3084#issuecomment-1732433168
    # Currently, I have been unable to set up Poetry to use GPU for Torch
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print("Set CUDA device")
    else:
        print("No CUDA device available, using CPU")

    model = YOLO("yolov8n.pt")

    # Depending on if the user wants to use a stream or a capture device,
    # Set the video capture to the appropriate source
    if args.url:
        video_capture = cv2.VideoCapture(args.url)
    else:
        video_capture = cv2.VideoCapture(args.capture_device)

    # Eliminate lag by setting the buffer size to 1
    # This makes it so that the video capture will only grab the most recent frame
    # However, this means that the video may be choppy
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Print the resolution of the video
    print(
        f"Video resolution: {video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)}x{video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)}"  # noqa: E501
    )

    print("Beginning video capture...")
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        # Only process every other frame of video to save time
        # Resize frame of video to a smaller size for faster recognition processing
        run_frame = cv2.resize(frame, (0, 0), fx=args.run_scale, fy=args.run_scale)
        # view_frame = cv2.resize(frame, (0, 0), fx=args.view_scale, fy=args.view_scale)

        results = model(run_frame, verbose=False)
        for i, r in enumerate(results):
            # list of dicts with each dict containing a label, x1, y1, x2, y2
            plot_boxes = []
            # Setup dictionary of object names
            if not object_names:
                for name in r.names.values():
                    object_names[name] = {
                        "last_detection_time": None,
                        "detection_duration": None,
                        # "first_detection_time": None,
                        "last_notification_time": None,
                    }
                # Also, make sure that the objects to detect are in the list of object_names
                # If it isn't, print a warning
                for obj in args.detect_object:
                    if obj not in object_names:
                        print(
                            f"Warning: {obj} is not in the list of objects the model can detect!"
                        )

            for box in r.boxes:
                # Get the name of the object
                class_id = r.names[box.cls[0].item()]
                # Get the coordinates of the object
                cords = box.xyxy[0].tolist()
                cords = [round(x) for x in cords]
                # Get the confidence
                conf = round(box.conf[0].item(), 2)
                # Print it out, adding a spacer between each object
                # print("Object type:", class_id)
                # print("Coordinates:", cords)
                # print("Probability:", conf)
                # print("---")

                # Now do stuff (if conf > 0.5)
                if conf < args.confidence_threshold or (
                    class_id not in args.detect_object and args.detect_object != []
                ):
                    # If the confidence is too low
                    # or if the object is not in the list of objects to detect and the list of objects to detect is not empty
                    # then skip this iteration
                    continue

                # Add the object to the list of objects to plot
                plot_boxes.append(
                    {
                        "label": class_id,
                        "x1": cords[0],
                        "y1": cords[1],
                        "x2": cords[2],
                        "y2": cords[3],
                    }
                )

                # End goal: Send a notification when an object has been detected for 2 seconds in the past 15 seconds.
                # However, don't send a notification if the last notification was less than 15 seconds ago

                # (re)start cycle
                if (
                    # If the object has not been detected before
                    object_names[class_id]["last_detection_time"] is None
                    # If the last detection was more than 15 seconds ago
                    or time.time() - object_names[class_id]["last_detection_time"]
                    > args.detection_window
                ):
                    # Set the last detection time to now
                    object_names[class_id]["last_detection_time"] = time.time()
                    print(f"First detection of {class_id} in this detection window")
                    # This line is important. It resets the detection duration when the object hasn't been detected for a while
                    # If detection duration is None, don't print anything.
                    # Otherwise, print that the detection duration is being reset due to inactivity
                    if object_names[class_id]["detection_duration"] is not None:
                        print(
                            f"Resetting detection duration for {class_id} since it hasn't been detected for {args.detection_window} seconds"  # noqa: E501
                        )
                    object_names[class_id]["detection_duration"] = 0
                else:
                    # Check if the last notification was less than 15 seconds ago
                    # If it was, then don't do anything
                    if (
                        time.time() - object_names[class_id]["last_detection_time"]
                        <= args.notification_window
                    ):
                        pass
                    # If it was more than 15 seconds ago, reset the detection duration
                    # This effectively resets the notification timer
                    else:
                        print("Notification timer has expired - resetting")
                        object_names[class_id]["detection_duration"] = 0
                    object_names[class_id]["detection_duration"] += (
                        time.time() - object_names[class_id]["last_detection_time"]
                    )
                    # print("Updating detection duration")
                    object_names[class_id]["last_detection_time"] = time.time()

                # (re)send notification
                # Check if detection has been ongoing for 2 seconds or more in the past 15 seconds
                if (
                    object_names[class_id]["detection_duration"]
                    >= args.detection_duration
                    and time.time() - object_names[class_id]["last_detection_time"]
                    <= args.detection_window
                ):
                    # If the last notification was more than 15 seconds ago, then send a notification
                    if (
                        object_names[class_id]["last_notification_time"] is None
                        or time.time()
                        - object_names[class_id]["last_notification_time"]
                        > args.notification_window
                    ):
                        object_names[class_id]["last_notification_time"] = time.time()
                        print(
                            f"Detected {class_id} for {args.detection_duration} seconds"
                        )
                        headers = notify.construct_ntfy_headers(
                            title=f"{class_id} detected",
                            tag="rotating_light",
                            priority="default",
                        )
                        notify.send_notification(
                            data=f"{class_id} detected for {args.detection_duration} seconds",
                            headers=headers,
                            url=args.ntfy_url,
                        )
                        # Reset the detection duration
                        print("Just sent a notification - resetting detection duration")
                        object_names[class_id]["detection_duration"] = 0
            frame_to_show = utils.plot_label(
                boxes=plot_boxes,
                full_frame=frame,
                # full_frame=r.plot(),
                run_scale=args.run_scale,
                view_scale=args.view_scale,
            )

            # Display the resulting frame
            # cv2.imshow("", r)
            cv2.imshow(f"Video{i}", frame_to_show)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release handle to the webcam
    print("Releasing video capture")
    video_capture.release()
    cv2.destroyAllWindows()


main()