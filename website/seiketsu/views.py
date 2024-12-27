from django.shortcuts import render
from .src.contopdetector import ContopDetector
from django.http import StreamingHttpResponse, Http404
from django.contrib.staticfiles import finders

CAMERA_NAMES = [
    "FREEMETAL1",
    "FREEMETAL2",
    "METALDET1",
    "FOLDING1",
    "FOLDING2",
    "FOLDING3",
    "SEWINGBACK2",
]


def index(request):
    cameras = CAMERA_NAMES
    return render(request, "fives/index.html", {"cameras": cameras})


def seiketsu_feed(request, camera_name):
    if camera_name not in CAMERA_NAMES:
        raise Http404("Camera not found")

    detector = ContopDetector(
        camera_name=camera_name,
        # video_source=finders.find("videos/contop testing.mp4"),
    )
    return StreamingHttpResponse(detector.stream_frames(), content_type="multipart/x-mixed-replace; boundary=frame")


def show_realtime_seiketsu(request):  # ini belum digunakan
    return render(request, "core/index.html")
