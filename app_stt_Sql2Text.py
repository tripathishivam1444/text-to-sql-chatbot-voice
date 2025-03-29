import streamlit as st
import pandas as pd
import openai
import os
import tempfile
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import pygame
from utils import create_Xsql_agent, clear_database, upload_to_database, table_exists
import re

# OpenAI API Configuration
openai.api_key = os.environ.get("OPENAI_API_KEY", "")

# Set Streamlit page configuration
st.set_page_config(
    page_title="Travel Database Chatbot",
    page_icon="🧳",
    initial_sidebar_state = "collapsed"
)


col1, col2 = st.columns([7, 1])

with col1:
    st.write("")

with col2:
    st.image("logo.png")

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'agent' not in st.session_state:
    st.session_state.agent = create_Xsql_agent() if table_exists() else None

if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False

if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False

# Function to play audio using pygame
def play_audio(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    st.session_state.is_playing = True

# Function to stop audio playback
def stop_audio():
    if st.session_state.is_playing:
        pygame.mixer.music.stop()
        st.session_state.is_playing = False

# Function to convert text to speech using OpenAI
def text_to_speech(text):
    try:
        response = openai.audio.speech.create(
            model="tts-1",
            voice="nova",  # Other voice options: "alloy", "nova", "echo", "shimmer"
            input=text
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(response.content)
            return temp_audio.name
    except Exception as e:
        st.error(f"Error with text-to-speech: {str(e)}")
        return None

# Function to record audio with amplification
def record_audio(duration=8, samplerate=44100, amplification_factor=18.0):
    st.info(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.float32)
    sd.wait()  # Wait for recording to complete

    # Amplify and clip audio
    amplified_audio = np.clip(audio * amplification_factor, -1.0, 1.0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        wav.write(temp_audio.name, samplerate, (amplified_audio * 32767).astype(np.int16))
        return temp_audio.name

# Function to transcribe speech to text using OpenAI
def transcribe_audio(audio_file):
    try:
        with open(audio_file, "rb") as file:
            transcript = openai.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=file
            )
        return transcript.text
    except Exception as e:
        st.error(f"Error with speech-to-text: {str(e)}")
        return ""

import re

def extract_valid_output(output):
    # st.write("this is from extract_valid_output function ----> ", output)
    error_indicator = "Could not parse LLM output:"
    if error_indicator in output:
        match = re.search(r"Could not parse LLM output:(.*)", output, re.S)
        if match:
            return match.group(1).strip()  # Return extracted content
    return output  # Return original output if no error is found



# Sidebar for Data Management
with st.sidebar:
    st.title("Data Management")

    if not table_exists():
        uploaded_file = st.file_uploader("Upload Travel Data (CSV or Excel)", type=['csv', 'xlsx'])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                st.write("Data Preview:")
                st.dataframe(df.head(3), use_container_width=True)

                if st.button("Upload to Database"):
                    with st.spinner("Uploading data to database..."):
                        success, message = upload_to_database(df)
                        if success:
                            st.session_state.agent = create_Xsql_agent()
                            st.success(message)
                        else:
                            st.error(message)

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    if st.button("Clear Database"):
        with st.spinner("Clearing database..."):
            success, message = clear_database()
            if success:
                st.session_state.agent = None
                st.session_state.chat_history = []
                st.success(message)
            else:
                st.error(message)

# Main Interface
st.title("🧳 Travel Database Chatbot")
st.subheader("Ask about travel destinations")

# Display Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Stop Button to Stop Ongoing Speech
if st.button("🔇 Stop Voice"):
    stop_audio()

# Input Mode (Text Input or Voice Input)
mode = st.radio("Input Mode", ["Text Input", "🎤 Mic Input"])

prompt = None


# Process user input and get a response
def handle_agent_response(prompt):
    prompt+= "LIMIT 15"
    try:
        with st.spinner("Thinking..."):
            if st.session_state.agent is None:
                st.error("Agent not initialized. Please upload data first.")
                return

            result = st.session_state.agent.invoke(prompt, handle_parsing_errors=True)

            full_response = result.get('output', 'No response generated')
            
            extracted_response = extract_valid_output(str(full_response))

            st.markdown(extracted_response)

            audio_path = text_to_speech(extracted_response)
            if audio_path:
                play_audio(audio_path)

            st.session_state.chat_history.append({"role": "assistant", "content": extracted_response})

    except Exception as e:
        # st.error(f"Error: {str(e)}")

        extracted_response = extract_valid_output(str(e))

        st.markdown(extracted_response)

        audio_path = text_to_speech(extracted_response)
        if audio_path:
            play_audio(audio_path)
        st.session_state.chat_history.append({"role": "assistant", "content": extracted_response})


# Handle Mic Input
if mode == "🎤 Mic Input":
    if not st.session_state.is_recording:
        if st.button("🎙️ Start Recording"):
            st.session_state.is_recording = True
            st.session_state.audio_path = record_audio(duration=8)
            st.session_state.is_recording = False

            prompt = transcribe_audio(st.session_state.audio_path)

            # Add user input to chat history and process response
            if prompt:
                
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                handle_agent_response(prompt)

    else:
        if st.button("🛑 Stop Recording"):
            st.session_state.is_recording = False
            st.info("Recording stopped.")

# Handle Text Input
elif mode == "Text Input":
    prompt = st.chat_input("Ask a question about travel destinations...")

    if prompt:
        
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        handle_agent_response(prompt)
