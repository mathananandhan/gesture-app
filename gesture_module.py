# gesture_module.py

import cv2
import mediapipe as mp
from gtts import gTTS
import pygame
import os
import pywhatkit as kit
from streamlit_webrtc import VideoTransformerBase

# Initialize mixer once
pygame.mixer.init()

# Speak message using gTTS
def speak_message(message, lang="ta"):
    tts = gTTS(text=message, lang=lang)
    tts.save("output.mp3")
    pygame.mixer.music.load("output.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()
    os.remove("output.mp3")

# Send WhatsApp message
def send_whatsapp_message(message, phone):
    try:
        kit.sendwhatmsg_instantly(phone, message, wait_time=10, tab_close=True)
    except Exception as e:
        print(f"WhatsApp send error: {e}")

class GestureDetector(VideoTransformerBase):
    def __init__(self, phone, lang, log):
        self.hands = mp.solutions.hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
        self.drawer = mp.solutions.drawing_utils
        self.gesture_buffer = []
        self.buffer_size = 5
        self.last_sent = None
        self.phone = phone
        self.lang = lang
        self.log = log
        self.messages_dict = {
            1: {"ta": "எனக்கு உணவு வேண்டும்!", "en": "I want food!"},
            2: {"ta": "எனக்கு கழிவறைக்கு செல்ல வேண்டும்!", "en": "I need to go to the restroom!"},
            3: {"ta": "எனக்கு தண்ணீர் வேண்டும்!", "en": "I want water!"},
            4: {"ta": "உதவி தேவை!", "en": "Help me!"},
            5: {"ta": "இங்கே வாருங்கள்!", "en": "Come here!"},
        }

    def transform(self, frame):
        import streamlit as st  # Ensures late import to avoid cyclic issues
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        finger_count = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.drawer.draw_landmarks(img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                # Thumb
                if hand_landmarks.landmark[4].x < hand_landmarks.landmark[2].x:
                    finger_count += 1

                # Other fingers
                finger_count += sum(
                    1 for i in [8, 12, 16, 20]
                    if hand_landmarks.landmark[i].y < hand_landmarks.landmark[i - 2].y
                )

                # Buffer update
                self.gesture_buffer.append(finger_count)
                if len(self.gesture_buffer) > self.buffer_size:
                    self.gesture_buffer.pop(0)

                # Gesture confirmation
                if self.gesture_buffer.count(finger_count) == self.buffer_size:
                    if finger_count in self.messages_dict:
                        message = self.messages_dict[finger_count][self.lang]
                        if self.last_sent != message:
                            self.log.append(f"🖐️ Gesture: {message}")
                            speak_message(message, self.lang)
                            send_whatsapp_message(message, self.phone)
                            self.last_sent = message
                            self.gesture_buffer.clear()

        return img
