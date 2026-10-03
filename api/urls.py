from django.urls import path

from .views import UploadDocumentView, RAGQueryView

urlpatterns = [
    path('upload/', UploadDocumentView.as_view()),
    path('query/', RAGQueryView.as_view()),
]