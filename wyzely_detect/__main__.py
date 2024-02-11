# import face_recognition
from pathlib import Path
import cv2

from prettytable import PrettyTable

# import hjson as json
import torch
from ultralytics import YOLO

from .utils import utils
from .utils.cli_args import argparser

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
args = None


def main():
    global objects_and_peoples
    global args 
    # RUN_BY_COMPOSE = os.getenv("RUN_BY_COMPOSE") # Replace this with code to check for gpu

    args = argparser.parse_args()

    # Check if a CUDA GPU is available. If it is, set it via torch. If not, set it to cpu
    # https://github.com/ultralytics/ultralytics/issues/3084#issuecomment-1732433168
    # Currently, I have been unable to set up Poetry to use GPU for Torch
    for i in range(torch.cuda.device_count()):
        print(f'Using {torch.cuda.get_device_properties(i).name} for pytorch')
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
        tf.config.set_visible_devices([], 'GPU')
        if tf.config.experimental.list_logical_devices('GPU'):
            print('GPU disabled unsuccessfully')
        else:
            print("GPU disabled successfully")

    model = YOLO("yolov8n.pt")

    # Depending on if the user wants to use a stream or a capture device,
    # Set the video capture to the appropriate source
    if not args.rtsp_url and not args.capture_device:
        print("No stream or capture device set, defaulting to capture device 0")
        args.capture_device = [0]
    else:
        video_sources = {
            "streams": [cv2.VideoCapture(url) for url in args.rtsp_url],
            "devices": [cv2.VideoCapture(device) for device in args.capture_device],
        }

    # Eliminate lag by setting the buffer size to 1
    # This makes it so that the video capture will only grab the most recent frame
    # However, this means that the video may be choppy
    # Only do this for streams
    for stream in video_sources["streams"]:
        stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Print the resolution of the video
    print(
        f"Video resolution: {video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)}x{video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)}"  # noqa: E501
    )
    print
    print("Beginning video capture...")
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        frame_to_show = utils.process_footage(
            frame = frame,
            run_scale = args.run_scale,
            view_scale = args.view_scale,

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
        # Display the resulting frame
        # TODO: When multi-camera support is added, this needs to be changed to allow all feeds
        if not args.no_display:
            cv2.imshow("Video", frame_to_show)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release handle to the webcam
    print("Releasing video capture")
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
