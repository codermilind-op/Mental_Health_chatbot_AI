import json
import re
import random
import numpy as np
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder

# Load intents data from JSON file
with open('chatbot/data/intents.json') as file:
    intents = json.load(file)

# Prepare the data for training the model (if needed)
data = []
responses = []  # Initialize a list to hold responses for each pattern

for intent in intents['intents']:
    for pattern in intent['patterns']:
        data.append((pattern, intent['tag']))
        responses.append(intent['responses'])  # Collect responses

# Create DataFrame from intents data
df = pd.DataFrame(data, columns=['patterns', 'tag'])

# Add responses as a new column in the DataFrame
df['responses'] = responses

# Initialize tokenizer and label encoder
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['patterns'])

lbl_enc = LabelEncoder()
lbl_enc.fit(df['tag'])

# Load your trained model (adjust path accordingly)
model = load_model('chatbot/models/model.h5')  # Point to your model file


def generate_answer(pattern):
    text = []
    txt = re.sub('[^a-zA-Z\']', ' ', pattern)  # Clean input text
    txt = txt.lower()
    text.append(txt)

    x_test = tokenizer.texts_to_sequences(text)
    x_test = np.array(x_test).squeeze()
    x_test = pad_sequences([x_test], padding='post', maxlen=model.input_shape[1])

    y_pred = model.predict(x_test)
    y_pred = y_pred.argmax()

    tag = lbl_enc.inverse_transform([y_pred])[0]

    # Check if tag exists in df and get responses
    if tag in df['tag'].values:
        responses = df[df['tag'] == tag]['responses'].values[0]
        return random.choice(responses)  # Return a random response from the list

    return "Sorry, I don't understand."  # Fallback response


@csrf_exempt  # Disable CSRF for simplicity; consider using CSRF tokens in production
def chat_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_input = data.get('message')
        bot_response = generate_answer(user_input)  # Call your Python function here
        return JsonResponse({'response': bot_response})


def chat_page(request):e
    return render(request, 'chatbot/chat.html')  # Render the chat page template