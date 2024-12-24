from django.shortcuts import render
from .src.broomdetector import BroomDetector
from django.http import StreamingHttpResponse
from django.contrib.staticfiles import finders


def index(request):
    return render(request, "seiso/index.html")


def seiso_feed(request):
    detector = BroomDetector(
        camera_name="OFFICE1",
        # video_source=finders.find("videos/vid_seiso_test.mp4"),
    )
    return StreamingHttpResponse(detector.stream_frames(), content_type="multipart/x-mixed-replace; boundary=frame")


def show_realtime_seiso(request):  # ini belum digunakan
    return render(request, "core/index.html")
