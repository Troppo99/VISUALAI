from django.shortcuts import render

def index(request):
    context = {
        "title": "VisualAI",
        "heading": "Welcome to VisualAI",
    }
    return render(request, "index.html", context)