# Chatbot Project

## Overview

This project integrates a machine learning model with a Django-based chatbot user interface to create a **Mental Health Chatbot AI**. The chatbot aims to assist users in addressing their mental health concerns by providing supportive and informative responses. Utilizing the trained model from `Untitled11.ipynb`, this AI-driven chatbot engages users in conversation, offering guidance and resources tailored to their needs. The goal is to create a safe space for individuals seeking help with their mental health issues.
## Project Structure

- **Untitled11.ipynb**: This notebook contains the machine learning model implementation, including data preprocessing, model training, and evaluation. It serves as the backbone of the chatbot's functionality.
- **Django Chatbot UI**: A web-based interface built with Django that allows users to interact with the chatbot. Users can send messages and receive responses generated by the machine learning model.

## Features

- **Interactive Chat Interface**: Users can type messages and receive real-time responses.
- **Machine Learning Integration**: The chatbot utilizes a trained machine learning model to generate responses based on user input.
- **API Endpoint**: A RESTful API endpoint is created to handle incoming messages and return responses in JSON format.

## Installation

Clone the repository:
   ```bash
   git clone 'copy repo link'
   cd your-repo-name

 python -m venv venv
source venv/bin/activate  # On macOS/Linux
.\venv\Scripts\activate  # On Windows

pip install -r requirements.txt
python manage.py runserver