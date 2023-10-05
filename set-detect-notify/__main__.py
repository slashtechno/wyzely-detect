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

from .utils import notify, config_utils

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
    # argparser.add_argument(
    # '--view-scale',
    # # Set it to the env VIEW_SCALE if it isn't blank, otherwise set it to 0.75
    # default=os.environ['VIEW_SCALE'] if 'VIEW_SCALE' in os.environ and os.environ['VIEW_SCALE'] != '' else 0.75,  # noqa: E501
    # type=float,
    # help="The scale to view the detection at, default is 0.75",
    # )

    stream_source = argparser.add_mutually_exclusive_group()
    stream_source.add_argument(
        '--url',
        default=os.environ['URL'] if 'URL' in os.environ and os.environ['URL'] != '' else None,  # noqa: E501
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
        # view_frame = cv2.resize(frame, (0, 0), fx=`args.`view_scale, fy=args.view_scale)

        results = model(run_frame, verbose=False)
        for r in results:
            # Setup dictionary of object names
            if not object_names:
                for name in r.names.values():
                    object_names[name] = {
                        "last_detection_time": None,
                        "detection_duration": None,
                        # "first_detection_time": None,
                    }  
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

                # Now do stuff

                # If this is the first time the object has been detected
                # or if it has been more than 15 seconds since the last detection
                # reset the detection duration

                if (
                    object_names[class_id]["last_detection_time"] is None
                    or time.time() - object_names[class_id]["last_detection_time"] > 15
                    or object_names[class_id]["detection_duration"] is None
                ):
                    print(f"First detection of {class_id}")
                    # time.time() returns the number of seconds since the epoch
                    object_names[class_id]["last_detection_time"] = time.time()
                    # object_names[class_id]["first_detection_time"] = time.time()
                    object_names[class_id]["detection_duration"] = 0
                    headers = notify.construct_ntfy_headers(
                        title=f"{class_id} Detected",
                        tag="rotating_light",
                        priority="default",
                    )
                    notify.send_notification(
                        data=f"{class_id} Detected", headers=headers, url=args.ntfy_url
                    )
                else:
                    # Add the time since the last detection to the total detection duration
                    object_names[class_id]["detection_duration"] += (
                        time.time() - object_names[class_id]["last_detection_time"]
                    )


                # Check if detection has been ongoing for 2 seconds or more in the past 15 seconds
                if (
                    object_names[class_id]["detection_duration"] >= 2
                    and time.time() - object_names[class_id]["last_detection_time"]
                    <= 15
                ):
                    print(f"Detected {class_id} for 2 seconds")
                    headers = notify.construct_ntfy_headers(
                        title=f"{class_id} Detected",
                        tag="rotating_light",
                        priority="default",
                    )
                    notify.send_notification(
                        data=f"{class_id} Detected", headers=headers, url=args.ntfy_url
                    )
            im_array = r.plot()
            # Scale back up the coordinates of the locations of detected objects.
            # im_array = np.multiply(im_array, 1/args.run_scale)
            # print(type(im_array))
            # print(im_array)
            # exit()
            cv2.imshow("View", im_array)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release handle to the webcam
    print("Releasing video capture")
    video_capture.release()
    cv2.destroyAllWindows()


main()
