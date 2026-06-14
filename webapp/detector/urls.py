# webapp/detector/urls.py
from django.urls import path
from . import views

app_name = "detector"

urlpatterns = [
    path("",                views.upload_view,  name="upload"),
    path("result/<int:pk>/", views.result_view,  name="result"),
    path("history/",        views.history_view, name="history"),
]