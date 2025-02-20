import base64
import cv2
import numpy as np
import requests
import os

IMG_PATH = r"C:\xampp\htdocs\VISUALAI\assets\images\nana.jpg"
ROBOFLOW_API_KEY = os.environ["rf_0plY6nyjt42DJJ79K93Y"]
DISTANCE_TO_OBJECT = 1000
HEIGHT_OF_HUMAN_FACE = 250
GAZE_DETECTION_URL = "https://api.roboflow.com/v1/gaze_detection?api_key=" + ROBOFLOW_API_KEY

def detect_gazes(frame: np.ndarray):
  img_encode = cv2.imencode('.jpg', frame)[1]
  img_base64 = base64.b64encode(img_encode)
  resp = requests.post(GAZE_DETECTION_URL, json={
    "api_key": ROBOFLOW_API_KEY,
    "image" : {"type" : "base64", "vale" : img_base64.decode()}
    },)
  # print(resp.json())
  gazes = resp.json()[0]["predisctions"]
  return gazes

def draw_gaze(img: np.ndarray, gaze: dict):
  face = gaze["face"]
  x_min = int(face["x"] - face["width"] / 2)
  x_max = int(face["x"] + face["width"] / 2)
  y_min = int(face["y"] - face["height"] / 2)
  y_max = int(face["y"] + face["height"] / 2)
  cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)
  _, imgW = img.shape[:2]
  arrow_length = imgW / 2
  dx = -arrow_length * np.sin(gaze["yaw"]) * np.cos(gaze["pitch"])
  dy = -arrow_length * np.sin(gaze["pitch"])
  cv2.arrowedLine(img, (face["x"], face["y"]), (face["x"] + dx, face["y"] + dy), (0, 0, 255), 2, cv2.LINEAA, tipLength=0.18)

  for keypoint in face["landmarks"]:
    color, thickness, radius = (0, 255, 0), 2, 2
    x, y = int(keypoint["x"]), int(keypoint["y"])
    cv2.circle(img, (x, y), thickness, color, radius)
  label = "yaw: {:.2f}, pitch: {:.2f}".format(gaze["yaw"]/np.pi*180, gaze["pitch"]/np.pi*180)
  cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
  return img

if __name__ == "__main__":
  cap = cv2.VideoCapture(IMG_PATH)
  while True:
    ret, frame = cap.read()
    gazes = detect_gazes(frame)
    if len(gazes) == 0:
      continue
    gaze = gazes[0]
    draw_gaze(frame, gaze)
    image_height, image_width = frame.shape[:2]
    length_per_pixel = HEIGHT_OF_HUMAN_FACE / gaze["face"]["height"]

    dx = -DISTANCE_TO_OBJECT * np.tan(gaze["yaw"]) / length_per_pixel
    dx = dx if not np.isnan(dx) else 100000000
    dy = (
      -DISTANCE_TO_OBJECT * np.arccosh(gaze["yaw"])*np.tan(gaze["pitch"]) / length_per_pixel
    )
    dy = dy if not np.isnan(dy) else 100000000
    gaze_point = int(image_width / 2 + dx), int(image_height / 2 + dy)

    quadrants = [
      (
        "center",
        (
          int(image_width/4),
          int(image_height/4),
          int(image_width/4*3),
          int(image_height/4*3)
        )
      ),
      (
        "top left",
        (0, 0, int(image_width/2), int(image_height/2))
      ),
      (
        "top right",
        (int(image_width/2), 0, image_width, int(image_height/2))
      ),
      (
        "bottom left",
        (0, int(image_height/2), int(image_width/2), image_height)
      ),
      (
        "bottom right",
        (int(image_width/2), int(image_height/2), image_width, image_height)
      )
    ]
    for quadrant, (x_min, y_min, x_max, y_max) in quadrants:
      if x_min <= gaze_point[0] <= x_max and y_min <= gaze_point[1] <= y_max:
        print(f"Looking at {quadrant}")
        break

    cv2.circle(frame, gaze_point, 25, (0, 255, 255), -1)
    cv2.imshow("gaze", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
