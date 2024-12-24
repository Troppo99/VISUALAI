from django.shortcuts import render
from django.http import StreamingHttpResponse
from .src.contopdetector import ContopDetector


def index(request):
    return render(request, "core/index.html")


def seiso_feed(request):
    detector = ContopDetector(
        contop_confidence_threshold=0.5,
        camera_name="FREEMETAL1",
        video_source=None,
    )
    return StreamingHttpResponse(detector.stream_frames(), content_type="multipart/x-mixed-replace; boundary=frame")


def show_realtime_seiso(request): # ini belum digunakan
    return render(request, "core/index.html")
