import cv2
import numpy as np
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
    view_frame = cv2.resize(full_frame, (0, 0), fx=view_scale, fy=view_scale)
    for thing in boxes:
        cv2.rectangle(
            # Image
            view_frame,
            # Start point
            (int(thing["x1"] * (run_scale/view_scale)), int(thing["y1"] * (run_scale/view_scale))),
            # End point
            (int(thing["x2"] * (run_scale/view_scale)), int(thing["y2"] * (run_scale/view_scale))),
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
            (int(thing["x1"] * (run_scale/view_scale)), int(thing["y1"] * (run_scale/view_scale))),
            # Font
            font,
            # Font Scale
            1,
            # Color
            (0, 255, 0),
            # Thickness
            1
        )
    return view_frame