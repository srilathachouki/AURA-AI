# app.py

import streamlit as st
import json
import random
import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import requests

# Voice Imports
import speech_recognition as sr
import pyttsx3

# Animation Import
from streamlit_lottie import st_lottie

# Load environment variables
load_dotenv()

# --- TTS ENGINE INITIALIZATION ---
try:
    tts_engine = pyttsx3.init()
except Exception as e:
    st.error(f"Failed to initialize TTS engine: {e}")
    tts_engine = None

# --- INTENTS JSON ---
INTENTS_JSON = """
{
  "intents": [
    { "tag": "greeting", "patterns": ["hi", "hello", "hey", "how are you"], "responses": ["Hello! How can I assist you today? 🚀", "Hi there! What can I do for you?"] },
    { "tag": "goodbye", "patterns": ["bye", "see you later", "goodbye"], "responses": ["Goodbye! Have a great day! 👋", "See you later!"] },
    { "tag": "thanks", "patterns": ["thanks", "thank you", "helpful"], "responses": ["You're welcome! 😊", "Happy to help!"] },
    { "tag": "about", "patterns": ["who are you", "what are you"], "responses": ["I am a chatbot powered by Streamlit and Google's Gemini API! 🤖", "I'm a language model's interface. Ask me anything!"] },
    { "tag": "creator", "patterns": ["who made you", "who created you"], "responses": ["I was built using a project plan from Google's Gemini.", "My creator brought me to life using Python and a powerful suite of tools."] }
  ]
}
"""

# --- STYLING AND ANIMATION ---
def load_lottieurl(url: str):
    """Fetches a Lottie animation from a URL."""
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not load Lottie animation: {e}")
        return None

# Inject custom CSS for a vibrant, modern look
def local_css():
    css = """
    <style>
        /* Main background gradient */
        .stApp {
            background-image: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            color: white;
        }

        /* Chat messages styling (Glassmorphism) */
        [data-testid="stChatMessage"] {
            border-radius: 20px;
            padding: 1rem 1.5rem;
            margin-bottom: 1rem;
            background-color: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        }
        
        /* User message alignment */
        [data-testid="stChatMessageContent"] {
             color: #E0E0E0; /* Light grey text for better readability */
        }
        
        /* Make title and caption pop */
        h1, .st-caption {
            text-align: center;
            color: #FFFFFF !important;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        }

        /* Style the chat input */
        [data-testid="stChatInput"] {
            background-color: rgba(0, 0, 0, 0.2);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        
        [data-testid="stChatInput"] > div > div > textarea {
            color: white;
        }

        /* Microphone button style */
        [data-testid="stButton"] button {
            background-color: #7b2cbf;
            color: white;
            border: none;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            font-size: 24px;
            transition: all 0.3s ease;
        }
        [data-testid="stButton"] button:hover {
            background-color: #9d4edd;
            transform: scale(1.1);
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: rgba(15, 12, 41, 0.8);
            backdrop-filter: blur(5px);
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- VOICE HELPER FUNCTIONS ---
def text_to_speech(text):
    """Converts text to speech."""
    if tts_engine:
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            st.warning(f"Could not process text-to-speech: {e}")

def speech_to_text():
    """Listens for voice input and converts it to text."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now!")
        try:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
            st.info("Recognizing...")
            text = r.recognize_google(audio)
            return text
        except Exception:
            st.warning("Could not understand or process audio.")
        return None

# --- CORE LOGIC ---
@st.cache_data
def load_and_train_model():
    data = json.loads(INTENTS_JSON)
    patterns, tags, responses_dict = [], [], {}
    for intent in data['intents']:
        responses_dict[intent['tag']] = intent['responses']
        for pattern in intent['patterns']:
            patterns.append(pattern.lower())
            tags.append(intent['tag'])
    model = make_pipeline(TfidfVectorizer(stop_words='english'), LogisticRegression(max_iter=500, random_state=42))
    model.fit(patterns, tags)
    return model, responses_dict

def get_chatbot_response(user_input, model, responses_dict):
    user_input_lower = user_input.lower()
    probabilities = model.predict_proba([user_input_lower])[0]
    if max(probabilities) > 0.65:
        predicted_tag = model.predict([user_input_lower])[0]
        return random.choice(responses_dict[predicted_tag])
    try:
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = gemini_model.generate_content(user_input)
        return response.text
    except Exception as e:
        st.error(f"An error occurred with the Google Gemini API: {e}", icon="🚨")
        return "Sorry, I'm having connection issues."

# --- STREAMLIT UI ---
st.set_page_config(page_title="Voice Assistant", page_icon="🔮", layout="centered")

# Apply the custom CSS
local_css()

intent_model, intent_responses = load_and_train_model()

# Sidebar with 3D animation
with st.sidebar:
    lottie_url = "https://assets10.lottiefiles.com/packages/lf20_v1yudlrx.json" # Floating robot
    lottie_animation = load_lottieurl(lottie_url)
    if lottie_animation:
        st_lottie(lottie_animation, speed=1, width=250, height=250, key="robot")
    
    st.header("About This App")
    st.markdown("This is a voice-enabled chatbot that uses local intents and the Google Gemini API to answer your questions.")
    st.markdown("---")
    st.markdown("Built with ❤️ using [Streamlit](https://streamlit.io).")
    st.markdown("By Chouki Srilatha")

st.title("🔮 AURA AI")
st.caption("Your friendly AI assistant. Try typing or using the mic.")

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    st.error("Your Google API key is not set! Please add it to your `.env` file.", icon="🔑")
    st.stop()
else:
    genai.configure(api_key=google_api_key)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def process_and_display_chat(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_chatbot_response(prompt, intent_model, intent_responses)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    # Call the text-to-speech function
    text_to_speech(response)
    time.sleep(0.5)

if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = ""

# Input controls
col1, col2 = st.columns([7, 1])
with col1:
    prompt = st.chat_input("What's on your mind?", key="chat_input")
with col2:
    if st.button("🎤", help="Click to speak"):
        transcribed_text = speech_to_text()
        if transcribed_text:
            st.session_state.transcribed_text = transcribed_text
            st.rerun()

# Process input from either text or voice
if prompt:
    process_and_display_chat(prompt)
elif st.session_state.transcribed_text:
    transcribed_prompt = st.session_state.transcribed_text
    st.session_state.transcribed_text = "" # Clear after use
    process_and_display_chat(transcribed_prompt)