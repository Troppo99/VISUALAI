from django.shortcuts import render
from .src.differdetector import DifferDetector
from django.http import StreamingHttpResponse
from django.contrib.staticfiles import finders


def index(request):
    return render(request, "seiton/index.html")


def seiton_feed(request):
    detector = DifferDetector(
        camera_name="FREEMETAL1",
        # video_source=finders.find("videos/contop testing.mp4"),
    )
    return StreamingHttpResponse(detector.stream_frames(), content_type="multipart/x-mixed-replace; boundary=frame")


def show_realtime_seiton(request):  # ini belum digunakan
    return render(request, "core/index.html")
