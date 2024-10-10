from django.urls import path
from .views import chat_view, chat_page

urlpatterns = [

    path('chat/', chat_view, name='chat'),
    path('', chat_page, name='chat_page'),  # Render chat page at root URL of chatbot app
]