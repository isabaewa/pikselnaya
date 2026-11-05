from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from editor import views

urlpatterns = [
    path("", views.index, name="index"),
    path("index.html", views.index),
    path("download/<str:result_id>/", views.download_result, name="download"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
