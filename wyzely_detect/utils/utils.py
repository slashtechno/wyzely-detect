import cv2
import numpy as np
from pathlib import Path
from deepface import DeepFace

first_face_try = True


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
):
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
    dict contains the following keys: label, x1, y1, x2, y2
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
    except ValueError as e:
        if (
            str(e)
            == "Face could not be detected. Please confirm that the picture is a face photo or consider to set enforce_detection param to False."  # noqa: E501
        ):
            # print("No faces recognized") # For debugging
            return None
        else:
            raise e
    # Iteate over the dataframes
    for df in face_dataframes:
        # The last row is the highest confidence
        # So we can just grab the path from there
        # iloc = Integer LOCation
        path_to_image = Path(df.iloc[-1]["identity"])
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

    """
    Example dataframe, for reference
    identity  (path to image) | source_x | source_y | source_w | source_h | VGG-Face_cosine (pretty much the confidence \_('_')_/) 
    """
