import streamlit as st
from streamlit_webrtc import webrtc_streamer
from gesture_module import GestureDetector

# Streamlit UI
st.title("✋ Silent Signals: Gesture to Speech & WhatsApp")
phone = st.text_input("📱 Enter WhatsApp Number (with country code):", "+91XXXXXXXXXX")
lang = st.selectbox("🈯 Select Language:", ["ta", "en"])
run = st.checkbox("✅ Start Detection")

# Log messages
log = st.empty()
messages_log = []

if run and phone:
    webrtc_streamer(
        key="gesture-detection",
        video_transformer_factory=lambda: GestureDetector(phone, lang, messages_log),
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )

    with log.container():
        st.subheader("📝 Gesture Log")
        for m in messages_log[::-1]:
            st.write(m)
else:
    st.info("Enter phone number and enable the checkbox to start gesture detection.")