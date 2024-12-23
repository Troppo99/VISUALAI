from django.shortcuts import render


def index(request):
    context = {
        "title": "VisualAI",
        "heading": "This is AI Streaming",
    }
    return render(request, "stream/index.html", context)
