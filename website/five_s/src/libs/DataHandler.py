import pymysql
import cvzone
from datetime import datetime
import os
import cv2


class DataHandler:
    def __init__(self, host="10.5.0.2", user="robot", password="robot123", database="visualai_db", table="cleaning", port=3307, task=""):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.table = table
        self.port = port
        self.connection = None
        self.cursor = None
        self.image_path = None
        self.task = task

    def config_database(self):
        try:
            self.connection = pymysql.connect(host=self.host, user=self.user, password=self.password, database=self.database, port=self.port)
            self.cursor = self.connection.cursor()
        except pymysql.MySQLError as e:
            print(f"Database connection failed: {e}")
            raise

    def save_data(self, frame, percentage, camera_name, insert=True):
        try:
            cvzone.putTextRect(frame, f"Datetime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", (10, 30), scale=1, thickness=2, offset=5)
            cvzone.putTextRect(frame, f"Camera: {camera_name}", (10, 90), scale=1, thickness=2, offset=5)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.image_path = rf"C:\xampp\htdocs\VISUALAI\website\media\brooming_frames\{camera_name}_{timestamp_str}.jpg"
            os.makedirs(os.path.dirname(self.image_path), exist_ok=True)
            cv2.imwrite(self.image_path, frame)
            if insert:
                self.insert_data(percentage)
            print("Image saved and inserted successfully" if insert else "Image saved without inserting")
        except Exception as e:
            print(f"Failed to save image: {e}")
            raise

    def insert_data(self, percentage):
        try:
            self.config_database()
            if not self.image_path:
                raise ValueError("Image path is not set")

            with open(self.image_path, "rb") as file:
                binary_image = file.read()

            camera_name = os.path.basename(self.image_path).split("_")[0]
            camera_name = camera_name + self.task
            query = f"""
            INSERT INTO {self.table} (camera_name, percentage, image)
            VALUES (%s, %s, %s)
            """
            self.cursor.execute(query, (camera_name, percentage, binary_image))
            self.connection.commit()
        except Exception as e:
            print(f"Error saving data: {e}")
        finally:
            if self.connection:
                self.connection.close()
