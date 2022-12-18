import datetime
import face_recognition
import cv2
import numpy as np
from dotenv import load_dotenv
import os
import json
import pathlib
import requests
import time


load_dotenv()
URL = os.getenv("URL")
RUN_SCALE = os.getenv("RUN_SCALE")
VIEW_SCALE = os.getenv("VIEW_SCALE")
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
# RUN_SCALE = 0.25
# VIEW_SCALE = 0.75
DISPLAY = False
RUN_BY_COMPOSE = os.getenv("RUN_BY_COMPOSE")


def find_face_from_name(name):
    for face in config["faces"]:
        if config["faces"][face]["name"] == name:
            return face
    return None


def write_config():
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=4)


print("Hello, world!")

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
known_face_encodings = []
known_face_names = []
process_this_frame = True

# Load the config file, if it does not exist or is blank, create it
config = {
    # If RUN_BY_COMPOSE is true, set url to rtsp://wyze-bridge:8554/wyze_cam_name, otherwise set it to "rtsp://localhost:8554/wyze_cam_name"
    "URL": "rtsp://localhost:8554/wyze_cam_name" if not RUN_BY_COMPOSE else "rtsp://bridge:8554/wyze_cam_name",
    "run_scale": "0.25",
    "view_scale": "0.75",
    "faces": {
        "example1": {"image": "config/example1.jpg", "last_seen": ""},
        "example2": {"image": "config/example2.jpg", "last_seen": ""},
    },
    "display": True
}
config_path = pathlib.Path("config/config.json")
if config_path.exists():
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
else:
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=4)
    print("Config file created, please edit it and restart the program")
    print("For relative paths, use the format config/example.jpg")
    exit()


if URL:
    config["URL"] = URL
else:
    URL = config["URL"]
if RUN_SCALE:
    config["RUN_SCALE"] = RUN_SCALE
else:
    RUN_SCALE = float(config["RUN_SCALE"])
if VIEW_SCALE:
    config["VIEW_SCALE"] = VIEW_SCALE
else:
    VIEW_SCALE = float(config["VIEW_SCALE"])
if DISPLAY:
    config["DISPLAY"] = DISPLAY
else:
    DISPLAY = config["display"]

print(f"Current config: {config}")

# Try this 5 times, 5 seconds apart. If the stream is not available, exit
# for i in range(5):
#     # Check if HLS stream is available using the requests library
#     # If it is not, print an error and exit
#     url = URL.replace("rtsp", "http").replace(":8554", ":8888")
#     print(f"Checking if HLS stream is available at {url}...")
#     try:
#     #    Replace rtsp with http and the port with 8888
#         r = requests.get(url)
#         if r.status_code != 200:
#             print("HLS stream not available, please check your URL")
#             exit()
#     except requests.exceptions.RequestException as e:
#         print("HLS stream not available, please check your URL")
#         if i == 4:
#             exit()
#         else: 
#             print(f"Retrying in 5 seconds ({i+1}/5)")
#             time.sleep(5)
#             continue



for face in config["faces"]:
    # Load a sample picture and learn how to recognize it.
    image = face_recognition.load_image_file(config["faces"][face]["image"])
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
    # Append the key to the list of known face names
    known_face_names.append(face)

video_capture = cv2.VideoCapture(URL)
# Eliminate lag by setting the buffer size to 1
# This makes it so that the video capture will only grab the most recent frame
# However, this means that the video may be choppy
video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Print the resolution of the video
print(f"Video resolution: {video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)}x{video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    # if process_this_frame:
    if True:
        # Resize frame of video to a smaller size for faster face recognition processing
        run_frame = cv2.resize(frame, (0, 0), fx=RUN_SCALE, fy=RUN_SCALE)
        view_frame = cv2.resize(frame, (0, 0), fx=VIEW_SCALE, fy=VIEW_SCALE)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_run_frame = run_frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        # model cnn is gpu accelerated, but hog is cpu only
        face_locations = face_recognition.face_locations(rgb_run_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_run_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding
            )
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding
            )
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                # print("For debugging, I found a face!!!! :D this should not be included in the final product lol :P")
                name = known_face_names[best_match_index]
                last_seen = config["faces"][name]["last_seen"]
                # If it's never been seen, set the last seen time to six seconds ago so it will be seen
                # Kind of a hacky way to do it, but it works... hopefully
                if last_seen == "":
                    print(f"{name} has been seen")
                    config["faces"][name]["last_seen"] = (
                        datetime.datetime.now() - datetime.timedelta(seconds=6)
                    ).strftime(DATETIME_FORMAT)
                    write_config()
                # Check if the face has been seen in the last 5 seconds
                if datetime.datetime.now() - datetime.datetime.strptime(
                    last_seen, DATETIME_FORMAT
                ) > datetime.timedelta(seconds=5):
                    print(f"{name} has been seen")
                # Update the last seen time
                config["faces"][name]["last_seen"] = datetime.datetime.now().strftime(
                    DATETIME_FORMAT
                )
                write_config()
            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    # Iterate over each face found in the frame to draw a box around it
    # Zip is used to iterate over two lists at the same time
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top = int(top * (VIEW_SCALE / RUN_SCALE))
        right = int(right * (VIEW_SCALE / RUN_SCALE))
        bottom = int(bottom * (VIEW_SCALE / RUN_SCALE))
        left = int(left * (VIEW_SCALE / RUN_SCALE))

        # Draw a box around the face
        cv2.rectangle(view_frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(
            view_frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
        )
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(
            view_frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1
        )

    # Display the resulting image if DISPLAY is set to true
    if config["display"]:
        cv2.imshow("Scaled View", view_frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
