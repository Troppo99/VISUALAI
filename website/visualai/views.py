from django.shortcuts import render
from django.http import StreamingHttpResponse, Http404
from five_s.src.contopdetector import ContopDetector

CAMERA_NAMES = [
    "FREEMETAL1",
    "FREEMETAL2",
    "METALDET1",
    "FOLDING1",
    "FOLDING2",
    "FOLDING3",
    "CUTTING3",
]


def streaming_index(request):
    cameras = CAMERA_NAMES
    return render(request, "index.html", {"cameras": cameras})  # atau template lain


def camera_feed(request, camera_name):
    if camera_name not in CAMERA_NAMES:
        raise Http404("Camera not found")
    detector = ContopDetector(camera_name=camera_name)
    return StreamingHttpResponse(detector.stream_frames(), content_type="multipart/x-mixed-replace; boundary=frame")
