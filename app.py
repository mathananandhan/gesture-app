import streamlit as st
import cv2
import mediapipe as mp
from gtts import gTTS
import os
import pygame
import pywhatkit as kit
import time
import pandas as pd
from datetime import datetime
import numpy as np
from deepface import DeepFace
import pyautogui
from googletrans import Translator

# ------------------- Page Setup -------------------
st.set_page_config(page_title="🖐️ Silent Signals", layout="wide")
st.title("🖐️ Silent Signals")
st.caption("AI-Powered Gesture + Emotion Detection | Multilingual Voice + WhatsApp Alerts")

# ------------------- Init State -------------------
if "detecting" not in st.session_state:
    st.session_state.detecting = False

# Initialize translator
translator = Translator()

# ------------------- Audio Init -------------------
try:
    pygame.mixer.init()
except Exception:
    st.warning("⚠️ Audio not initialized properly.")

# ------------------- Multilingual Language Codes -------------------
lang_code_map = {
    "Tamil": "ta",
    "English": "en",
    "Hindi": "hi"
}

# ------------------- Default Messages (can be in any language, will be translated dynamically) -------------------
base_messages = {
    1: "I need food!",
    2: "I need to go to the toilet!",
    3: "I need water!",
    4: "Help needed!",
    5: "Come here!"
}

base_emotion_alert = "Urgent emotion: {}"

# ------------------- Tamil Voice + Other Languages -------------------
def speak_message(message, lang):
    try:
        tts = gTTS(text=message, lang=lang)
        tts.save("output.mp3")
        pygame.mixer.music.load("output.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        os.remove("output.mp3")
    except Exception as e:
        st.error(f"Audio error: {e}")

# ------------------- WhatsApp Messaging -------------------
def send_whatsapp_message(message, phone):
    try:
        if not (phone.startswith("+") and len(phone) > 10 and phone[1:].replace(' ', '').replace('-', '').isdigit()):
            st.error("❌ Invalid phone number format.")
            return
        st.info("⏳ Sending WhatsApp message. Please don't touch the mouse or keyboard...")
        kit.sendwhatmsg_instantly(phone, message, wait_time=20, tab_close=True, close_time=10)
        time.sleep(8)  # Wait for WhatsApp Web to load and message to be typed
        pyautogui.press("enter")  # Send message
        st.success(f"✅ Message sent to {phone}: {message}")
    except Exception as e:
        st.error(f"❌ WhatsApp error: {e}")

# ------------------- Logging -------------------
def log_gesture_action(gesture_type, message):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = {"Time": now, "Gesture": gesture_type, "Message": message}
    if not os.path.isfile("gesture_log.csv"):
        pd.DataFrame([new_row]).to_csv("gesture_log.csv", index=False)
    else:
        df = pd.read_csv("gesture_log.csv")
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv("gesture_log.csv", index=False)

# ------------------- DeepFace Emotion Classifier -------------------
def classify_emotion_from_face(face_roi):
    try:
        temp_path = "temp_face.jpg"
        cv2.imwrite(temp_path, face_roi)
        result = DeepFace.analyze(img_path=temp_path, actions=['emotion'], enforce_detection=False)
        os.remove(temp_path)
        return result[0]['dominant_emotion'].capitalize()
    except Exception as e:
        st.warning(f"Emotion detection failed: {e}")
        return "Unknown"

# ------------------- Translation Helper -------------------
def translate_text(text, dest_lang):
    try:
        translated = translator.translate(text, dest=dest_lang)
        return translated.text
    except Exception as e:
        st.warning(f"Translation failed: {e}")
        return text

# ------------------- Main Detection and Action -------------------
def detect_gesture_and_act(phone_number, action_choice, use_emotion, language_choice):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
    mp_draw = mp.solutions.drawing_utils
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)

    cap = cv2.VideoCapture(0)
    gesture_buffer = []
    buffer_size = 5
    detected_finger = None
    detected_emotion = None

    video_placeholder = st.empty()
    st.info("📷 Camera is active. Show hand and face to detect gestures/emotions.")

    lang_code = lang_code_map.get(language_choice, "en")

    while st.session_state.detecting:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Camera error.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        face_results = face_mesh.process(rgb)

        # Gesture Detection
        finger_count = 0
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                thumb_tip = hand.landmark[4]
                thumb_base = hand.landmark[2]
                if thumb_tip.x < thumb_base.x:
                    finger_count += 1
                finger_count += sum(
                    1 for i in [8, 12, 16, 20] if hand.landmark[i].y < hand.landmark[i - 2].y
                )

                gesture_buffer.append(finger_count)
                if len(gesture_buffer) > buffer_size:
                    gesture_buffer.pop(0)

                if gesture_buffer.count(finger_count) == buffer_size and finger_count != detected_finger:
                    detected_finger = finger_count
                    base_msg = base_messages.get(finger_count)
                    if base_msg:
                        translated_msg = translate_text(base_msg, lang_code)
                        st.success(f"✅ Detected: {translated_msg}")
                        if action_choice == "🔊 Voice Only":
                            speak_message(translated_msg, lang_code)
                        elif action_choice == "💬 WhatsApp Only":
                            send_whatsapp_message(translated_msg, phone_number)
                        elif action_choice == "✅ Both":
                            speak_message(translated_msg, lang_code)
                            send_whatsapp_message(translated_msg, phone_number)
                        log_gesture_action(finger_count, translated_msg)

                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # Emotion Detection
        if use_emotion and face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                ih, iw, _ = frame.shape
                x_coords = [int(lm.x * iw) for lm in face_landmarks.landmark]
                y_coords = [int(lm.y * ih) for lm in face_landmarks.landmark]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                face_roi = frame[y_min:y_max, x_min:x_max]
                if face_roi.size > 0:
                    emotion = classify_emotion_from_face(face_roi)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                    cv2.putText(frame, emotion, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                    if emotion in ["Angry", "Fear", "Sad"] and emotion != detected_emotion:
                        detected_emotion = emotion
                        base_emotion_msg = base_emotion_alert.format(emotion)
                        translated_emotion_msg = translate_text(base_emotion_msg, lang_code)
                        st.error(f"⚠️ Emotion Alert: {translated_emotion_msg}")
                        if action_choice == "🔊 Voice Only":
                            speak_message(translated_emotion_msg, lang_code)
                        elif action_choice == "💬 WhatsApp Only":
                            send_whatsapp_message(translated_emotion_msg, phone_number)
                        elif action_choice == "✅ Both":
                            speak_message(translated_emotion_msg, lang_code)
                            send_whatsapp_message(translated_emotion_msg, phone_number)
                        log_gesture_action("Emotion", translated_emotion_msg)

        video_placeholder.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()

# ------------------- Streamlit UI -------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📲 Controls")

    phone_number = st.text_input("📞 WhatsApp Number", value="+91 ")

    language_choice = st.selectbox(
        "🌏 Select message language:",
        list(lang_code_map.keys())
    )

    action_choice = st.radio(
        "🎯 Choose Action:",
        ("🔊 Voice Only", "💬 WhatsApp Only", "✅ Both"),
        horizontal=True
    )

    use_emotion = st.toggle("🧠 Enable Emotion Detection", value=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶️ Start Detection"):
            if phone_number.startswith("+") and len(phone_number) >= 10:
                st.session_state.detecting = True
            else:
                st.error("❌ Invalid phone number.")

    with c2:
        if st.button("⏹ Stop Detection"):
            st.session_state.detecting = False

with col2:
    st.header("📜 Activity Log + Feed")
    if os.path.exists("gesture_log.csv"):
        df_log = pd.read_csv("gesture_log.csv")
        st.dataframe(df_log, use_container_width=True)
        csv = df_log.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Log", data=csv, file_name="gesture_log.csv", mime="text/csv")
        if st.button("🧹 Clear Log"):
            os.remove("gesture_log.csv")
            st.success("🗑️ Log cleared.")
    else:
        st.info("No gesture or emotion log available.")

# ------------------- Run detection -------------------
if st.session_state.detecting:
    detect_gesture_and_act(phone_number, action_choice, use_emotion, language_choice)

