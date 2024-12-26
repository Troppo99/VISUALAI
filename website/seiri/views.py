from django.shortcuts import render
from .src.conedetector import ConeDetector
from django.http import StreamingHttpResponse
from django.contrib.staticfiles import finders


def index(request):
    return render(request, "seiri/index.html")


def seiri_feed(request):
    detector = ConeDetector(
        camera_name="FREEMETAL1",
        # video_source=finders.find("videos/contop testing.mp4"),
    )
    return StreamingHttpResponse(detector.stream_frames(), content_type="multipart/x-mixed-replace; boundary=frame")


def show_realtime_seiri(request):  # ini belum digunakan
    return render(request, "core/index.html")
