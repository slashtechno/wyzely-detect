# import face_recognition
from pathlib import Path
import cv2
import sys
from prettytable import PrettyTable

# import hjson as json
import torch
from ultralytics import YOLO

from .utils import utils
from .utils.cli_args import argparser

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
args = None


def main():
    global args

    args = argparser.parse_args()

    # Check if a CUDA GPU is available. If it is, set it via torch. If not, set it to cpu
    # https://github.com/ultralytics/ultralytics/issues/3084#issuecomment-1732433168
    # Currently, I have been unable to set up Poetry to use GPU for Torch
    for i in range(torch.cuda.device_count()):
        print(f"Using {torch.cuda.get_device_properties(i).name} for pytorch")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print("Set CUDA device")
    else:
        print("No CUDA device available, using CPU")
    # Seems automatically, deepface (tensorflow) tried to use my GPU on Pop!_OS (I did not set up cudnn or anything)
    # Not sure the best way, in Poetry, to manage GPU libraries so for now, just use CPU
    if args.force_disable_tensorflow_gpu:
        print("Forcing tensorflow to use CPU")
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
        if tf.config.experimental.list_logical_devices("GPU"):
            print("GPU disabled unsuccessfully")
        else:
            print("GPU disabled successfully")

    model = YOLO("yolov8n.pt")

    # Depending on if the user wants to use a stream or a capture device,
    # Set the video capture to the appropriate source
    if not args.rtsp_url and not args.capture_device:
        print("No stream or capture device set, defaulting to capture device 0")
        video_sources = {"devices": [cv2.VideoCapture(0)]}
    else:
        video_sources = {
            "streams": [cv2.VideoCapture(url) for url in args.rtsp_url],
            "devices": [cv2.VideoCapture(device) for device in args.capture_device],
        }

    if args.fake_second_source:
        try:
            video_sources["devices"].append(video_sources["devices"][0])
        except KeyError:
            print("No capture device to use as second source. Trying stream.")
            try:
                video_sources["devices"].append(video_sources["devices"][0])
            except KeyError:
                print("No stream to use as a second source")
                # When the code tries to resize the nonexistent capture device 1, the program will fail

    # Eliminate lag by setting the buffer size to 1
    # This makes it so that the video capture will only grab the most recent frame
    # However, this means that the video may be choppy
    # Only do this for streams
    try:
        for stream in video_sources["streams"]:
            stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # If there are no streams, this will throw a KeyError
    except KeyError:
        pass

    # Print out the resolution of the video sources. Ideally, change this so the device ID/url is also printed
    pretty_table = PrettyTable(field_names=["Source Type", "Resolution"])
    for source_type, sources in video_sources.items():
        for source in sources:
            if (
                source.get(cv2.CAP_PROP_FRAME_WIDTH) == 0
                or source.get(cv2.CAP_PROP_FRAME_HEIGHT) == 0
            ):
                message = "Capture for a source failed as resolution is 0x0.\n"
                if source_type == "streams":
                    message += "Check if the stream URL is correct and if the stream is online."
                else:
                    message += "Check if the capture device is connected, working, and not in use by another program."
                print(message)
                sys.exit(1)
            pretty_table.add_row(
                [
                    source_type,
                    f"{source.get(cv2.CAP_PROP_FRAME_WIDTH)}x{source.get(cv2.CAP_PROP_FRAME_HEIGHT)}",
                ]
            )
    print(pretty_table)
    print("Beginning video capture...")
    while True:
        # Grab a single frame of video
        frames = []
        # frames = [source.read() for sources in video_sources.values() for source in sources]
        for list_of_sources in video_sources.values():
            frames.extend([source.read()[1] for source in list_of_sources])
        frames_to_show = []
        for frame in frames:
            frames_to_show.append(
                utils.process_footage(
                    frame=frame,
                    run_scale=args.run_scale,
                    view_scale=args.view_scale,
                    faces_directory=Path(args.faces_directory),
                    face_confidence_threshold=args.face_confidence_threshold,
                    no_remove_representations=args.no_remove_representations,
                    detection_window=args.detection_window,
                    detection_duration=args.detection_duration,
                    notification_window=args.notification_window,
                    ntfy_url=args.ntfy_url,
                    model=model,
                    detect_object=args.detect_object,
                    object_confidence_threshold=args.object_confidence_threshold,
                )
            )
        # Display the resulting frame
        if not args.no_display:
            for i, frame_to_show in enumerate(frames_to_show):
                cv2.imshow(f"Video {i}", frame_to_show)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release handle to the webcam
    print("Releasing video capture")
    [source.release() for sources in video_sources.values() for source in sources]
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
