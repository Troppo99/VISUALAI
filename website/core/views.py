from django.shortcuts import render
from django.http import StreamingHttpResponse
from .src.contopdetector import ContopDetector
from django.contrib.staticfiles import finders


def index(request):
    return render(request, "core/index.html")


def seiketsu_feed(request):
    detector = ContopDetector(
        contop_confidence_threshold=0.5,
        camera_name="FREEMETAL1",
        # video_source=finders.find("videos/contop testing.mp4"),
    )
    return StreamingHttpResponse(detector.stream_frames(), content_type="multipart/x-mixed-replace; boundary=frame")


def show_realtime_seiketsu(request): # ini belum digunakan
    return render(request, "core/index.html")
