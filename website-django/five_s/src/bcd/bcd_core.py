import cv2, cvzone, json, math, numpy as np, os, queue, threading, time, torch, sys
from shapely.geometry import Polygon
from shapely.ops import unary_union
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
sys.path.append(parent_dir)
from libs.DataHandler import DataHandler


class Detector:
    def __init__(self):
        pass

    def camera_config(self):
        pass

    def choose_video_source(self):
        pass

    def camera_capture(self):
        pass

    def export_frame(self):
        pass

    def process_frame(self):
        pass

    def main(self):
        print("anda masuk main")
        pass


if __name__ == "__main__":
    d = Detector()
    d.main()
