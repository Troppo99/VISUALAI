from django.shortcuts import render
from .src.contopdetector import ContopDetector
from django.http import StreamingHttpResponse
from django.contrib.staticfiles import finders


def index(request):
    video_count = 11
    videos = range(1, video_count + 1)
    return render(request, "seiketsu/index.html", {"videos": videos})


def seiketsu_feed(request):
    detector = ContopDetector(
        camera_name="FREEMETAL1",
        # video_source=finders.find("videos/contop testing.mp4"),
    )
    return StreamingHttpResponse(detector.stream_frames(), content_type="multipart/x-mixed-replace; boundary=frame")


def show_realtime_seiketsu(request):  # ini belum digunakan
    return render(request, "core/index.html")
