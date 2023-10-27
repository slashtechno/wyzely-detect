# import face_recognition
from pathlib import Path

import cv2

# import hjson as json
import torch
from ultralytics import YOLO

from .utils import notify, utils
from .utils.cli_args import argparser

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
args = None

objects_and_peoples = {
    "objects": {},
    "peoples": {},
}


def main():
    global objects_and_peoples
    global args
    # RUN_BY_COMPOSE = os.getenv("RUN_BY_COMPOSE") # Replace this with code to check for gpu

    args = argparser.parse_args()

    # Check if a CUDA GPU is available. If it is, set it via torch. If not, set it to cpu
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
    if args.rtsp_url is not None:
        video_capture = cv2.VideoCapture(args.rtsp_url)
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

        path_to_faces = Path(args.faces_directory)
        path_to_faces_exists = path_to_faces.is_dir()

        for i, r in enumerate(results):
            # list of dicts with each dict containing a label, x1, y1, x2, y2
            plot_boxes = []

            # The following is stuff for people
            # This is still in the for loop as each result, no matter if anything is detected, will be present.
            # Thus, there will always be one result (r)

            # Only run if path_to_faces exists
            # May be better to check every iteration, but this also works
            if path_to_faces_exists:
                if face_details := utils.recognize_face(
                    path_to_directory=path_to_faces,
                    run_frame=run_frame,
                    min_confidence=args.face_confidence_threshold,
                    no_remove_representations=args.no_remove_representations,
                ):
                    plot_boxes.append(face_details)
                    objects_and_peoples = notify.thing_detected(
                        thing_name=face_details["label"],
                        objects_and_peoples=objects_and_peoples,
                        detection_type="peoples",
                        detection_window=args.detection_window,
                        detection_duration=args.detection_duration,
                        notification_window=args.notification_window,
                        ntfy_url=args.ntfy_url,
                    )

            # The following is stuff for objects
            # Setup dictionary of object names
            if (
                objects_and_peoples["objects"] == {}
                or objects_and_peoples["objects"] is None
            ):
                for name in r.names.values():
                    objects_and_peoples["objects"][name] = {
                        "last_detection_time": None,
                        "detection_duration": None,
                        # "first_detection_time": None,
                        "last_notification_time": None,
                    }
                # Also, make sure that the objects to detect are in the list of objects_and_peoples
                # If it isn't, print a warning
                for obj in args.detect_object:
                    if obj not in objects_and_peoples:
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
                if conf < args.object_confidence_threshold or (
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

                objects_and_peoples = notify.thing_detected(
                    thing_name=class_id,
                    objects_and_peoples=objects_and_peoples,
                    detection_type="objects",
                    detection_window=args.detection_window,
                    detection_duration=args.detection_duration,
                    notification_window=args.notification_window,
                    ntfy_url=args.ntfy_url,
                )

            # To debug plotting, use r.plot() to cross reference the bounding boxes drawn by the plot_label() and r.plot()
            frame_to_show = utils.plot_label(
                boxes=plot_boxes,
                full_frame=frame,
                # full_frame=r.plot(),
                run_scale=args.run_scale,
                view_scale=args.view_scale,
            )

            # Display the resulting frame
            # cv2.imshow("", r)
            if not args.no_display:
                cv2.imshow(f"Video{i}", frame_to_show)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release handle to the webcam
    print("Releasing video capture")
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
