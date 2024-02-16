import cv2
import os
import numpy as np
from pathlib import Path

# https://stackoverflow.com/a/42121886/18270659
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


from deepface import DeepFace  # noqa: E402
from . import notify  # noqa: E402

first_face_try = True

# TODO: When multi-camera support is ~~added~~ improved, this will need to be changed so that each camera has its own dict
objects_and_peoples = {
    "objects": {},
    "peoples": {},
}


def process_footage(
    # Frame
    frame: np.ndarray = None,
    # scale
    run_scale: float = None,
    view_scale: float = None,
    # Face stuff
    faces_directory: str = None,
    face_confidence_threshold: float = None,
    no_remove_representations: bool = False,
    # Timer stuff
    detection_window: int = None,
    detection_duration: int = None,
    notification_window: int = None,
    ntfy_url: str = None,
    # Object stuff
    # YOLO object
    model=None,
    detect_object: list = None,
    object_confidence_threshold=None,
) -> np.ndarray:
    """Takes in a frame and processes it"""
    global objects_and_peoples

    # Resize frame of video to a smaller size for faster recognition processing
    run_frame = cv2.resize(frame, (0, 0), fx=run_scale, fy=run_scale)
    # view_frame = cv2.resize(frame, (0, 0), fx=args.view_scale, fy=args.view_scale)

    results = model(run_frame, verbose=False)

    path_to_faces = Path(faces_directory)
    path_to_faces_exists = path_to_faces.is_dir()

    for r in results:
        # list of dicts with each dict containing a label, x1, y1, x2, y2
        plot_boxes = []

        # The following is stuff for people
        # This is still in the for loop as each result, no matter if anything is detected, will be present.
        # Thus, there will always be one result (r)

        # Only run if path_to_faces exists
        # May be better to check every iteration, but this also works
        if path_to_faces_exists:
            if face_details := recognize_face(
                path_to_directory=path_to_faces,
                run_frame=run_frame,
                # Perhaps make these names match?
                min_confidence=face_confidence_threshold,
                no_remove_representations=no_remove_representations,
            ):
                plot_boxes.append(face_details)
                objects_and_peoples = notify.thing_detected(
                    thing_name=face_details["label"],
                    objects_and_peoples=objects_and_peoples,
                    detection_type="peoples",
                    detection_window=detection_window,
                    detection_duration=detection_duration,
                    notification_window=notification_window,
                    ntfy_url=ntfy_url,
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
            for obj in detect_object:
                # .keys() shouldn't be needed
                if obj not in objects_and_peoples["objects"]:
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
            if conf < object_confidence_threshold or (
                class_id not in detect_object and detect_object != []
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
                detection_window=detection_window,
                detection_duration=detection_duration,
                notification_window=notification_window,
                ntfy_url=ntfy_url,
            )

        # To debug plotting, use r.plot() to cross reference the bounding boxes drawn by the plot_label() and r.plot()
        frame_to_show = plot_label(
            boxes=plot_boxes,
            full_frame=frame,
            # full_frame=r.plot(),
            run_scale=run_scale,
            view_scale=view_scale,
        )
        # Unsure if this should also return the objects_and_peoples dict
        return frame_to_show


def plot_label(
    # list of dicts with each dict containing a label, x1, y1, x2, y2
    boxes: list = None,
    # opencv image
    full_frame: np.ndarray = None,
    # run_scale is the scale of the image that was used to run the model
    # So the coordinates will be scaled up to the view frame size
    run_scale: float = None,
    # view_scale is the scale of the image, in relation to the full frame
    # So the coordinates will be scaled appropriately when coming from run_frame
    view_scale: float = None,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
) -> np.ndarray:
    # x1 and y1 are the top left corner of the box
    # x2 and y2 are the bottom right corner of the box
    # Example scaling: full_frame: 1 run_frame: 0.5 view_frame: 0.25
    view_frame = cv2.resize(full_frame, (0, 0), fx=view_scale, fy=view_scale)
    for thing in boxes:
        cv2.rectangle(
            # Image
            view_frame,
            # Top left corner
            (
                int((thing["x1"] / run_scale) * view_scale),
                int((thing["y1"] / run_scale) * view_scale),
            ),
            # Bottom right corner
            (
                int((thing["x2"] / run_scale) * view_scale),
                int((thing["y2"] / run_scale) * view_scale),
            ),
            # Color
            (0, 255, 0),
            # Thickness
            2,
        )
        cv2.putText(
            # Image
            view_frame,
            # Text
            thing["label"],
            # Origin
            (
                int((thing["x1"] / run_scale) * view_scale),
                int((thing["y1"] / run_scale) * view_scale) - 10,
            ),
            # Font
            font,
            # Font Scale
            1,
            # Color
            (0, 255, 0),
            # Thickness
            1,
        )
    return view_frame


def recognize_face(
    path_to_directory: Path = Path("faces"),
    # opencv image
    run_frame: np.ndarray = None,
    min_confidence: float = 0.3,
    no_remove_representations: bool = False,
) -> np.ndarray:
    """
        Accepts a path to a directory of images of faces to be used as a refference
        In addition, accepts an opencv image to be used as the frame to be searched

        Returns a single dictonary as currently only 1 face can be detected in each frame
        Cosine threshold is 0.3, so if the confidence is less than that, it will return None
        dict conta                # Maybe use os.exit() instead?
    ins the following keys: label, x1, y1, x2, y2
        The directory should be structured as follows:
        faces/
            name/
                image1.jpg
                image2.jpg
                image3.jpg
            name2/
                image1.jpg
                image2.jpg
                image3.jpg
        (not neccessarily jpgs, but you get the idea)

        Point is, `name` is the name of the person in the images in the directory `name`
        That name will be used as the label for the face in the frame
    """
    global first_face_try

    # If it's the first time the function is being run, remove representations_arcface.pkl, if it exists
    if first_face_try and not no_remove_representations:
        try:
            path_to_directory.joinpath("representations_arcface.pkl").unlink()
            print("Removing representations_arcface.pkl")
        except FileNotFoundError:
            print("representations_arcface.pkl does not exist")
        first_face_try = False
    elif first_face_try and no_remove_representations:
        print("Not attempting to remove representations_arcface.pkl")
        first_face_try = False

    # face_dataframes is a vanilla list of dataframes
    # It seems face_dataframes is empty if the face database (directory) doesn't exist. Seems to work if it's empty though
    # This line is here to prevent a crash if that happens. However, there is a check in __main__ so it shouldn't happen
    face_dataframes = []
    try:
        face_dataframes = DeepFace.find(
            run_frame,
            db_path=str(path_to_directory),
            # Problem with enforce_detection=False is that it will always (?) return a face, no matter the confidence
            # Thus, false-positives need to be filtered out
            enforce_detection=False,
            silent=True,
            # Could use VGG-Face, but whilst fixing another issue, ArcFace seemed to be slightly faster
            # I read somewhere that opencv is the fastest (but not as accurate). Could be changed later, but opencv seems to work well
            model_name="ArcFace",
            detector_backend="opencv",
        )
        '''
        Example dataframe, for reference
        identity  (path to image) | source_x | source_y | source_w | source_h | VGG-Face_cosine (pretty much the confidence \\_('_')_/) 
        '''
    except ValueError as e:
        if (
            str(e)
            == "Face could not be detected. Please confirm that the picture is a face photo or consider to set enforce_detection param to False."  # noqa: E501
        ):
            # print("No faces recognized") # For debugging
            return None
        elif (
            # Check if the error message contains "Validate .jpg or .png files exist in this path."
            "Validate .jpg or .png files exist in this path."
            in str(e)
        ):
            # If a verbose/silent flag is added, this should be changed to print only if verbose is true
            # print("No faces found in database")
            return None
        else:
            raise e
    # Iteate over the dataframes
    for df in face_dataframes:
        # The last row is the highest confidence
        # So we can just grab the path from there
        # iloc = Integer LOCation
        try:
            path_to_image = Path(df.iloc[-1]["identity"])
        # Seems this is caused when someone steps into frame and their face is detected but not recognized
        except IndexError:
            print("Face present but not recognized")
            continue
        # If the parent name is the same as the path to the database, then set label to the image name instead of the parent name
        if path_to_image.parent == Path(path_to_directory):
            label = path_to_image.name
        else:
            label = path_to_image.parent.name
        # Return the coordinates of the box in xyxy format, rather than xywh
        # This is because YOLO uses xyxy, and that's how plot_label expects
        # Also, xyxy is just the top left and bottom right corners of the box
        coordinates = {
            "x1": df.iloc[-1]["source_x"],
            "y1": df.iloc[-1]["source_y"],
            "x2": df.iloc[-1]["source_x"] + df.iloc[-1]["source_w"],
            "y2": df.iloc[-1]["source_y"] + df.iloc[-1]["source_h"],
        }
        # After some brief testing, it seems positive matches are > 0.3
        cosine_similarity = df.iloc[-1]["ArcFace_cosine"]
        if cosine_similarity < min_confidence:
            return None
        # label = "Unknown"
        to_return = dict(label=label, **coordinates)
        print(
            f"Cosine similarity: {cosine_similarity}, filname: {path_to_image.name}, to_return: {to_return}"
        )
        return to_return
    return None
