import os
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
from tensorflow.keras.models import load_model
from gtts import gTTS
import tempfile
import cv2
import base64
from io import BytesIO

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "best_emotion_model.keras"
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

ADVICE = {
    'Angry': "You seem angry. Try taking deep breaths and relaxing.",
    'Disgust': "You seem disgusted. Take a moment to refocus on something calming.",
    'Fear': "You appear fearful. Remember you are safe now.",
    'Happy': "You look very happy — enjoy the moment!",
    'Sad': "You look sad. It's okay — breathe and be kind to yourself.",
    'Surprise': "You look surprised — something unexpected happened?",
    'Neutral': "Your expression is neutral — calm and steady."
}

SONG_FILES = {
    'Angry': 'angry.mp3',
    'Disgust': 'disgust.mp3',
    'Fear': 'fear.mp3',
    'Happy': 'happy.mp3',
    'Sad': 'sad.mp3',
    'Surprise': 'surprise.mp3',
    'Neutral': 'neutral.mp3'
}

MAX_FILE_SIZE_MB = 5  # Limit uploads to 5MB
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# -------------------------
# PAGE SETTINGS
# -------------------------
st.set_page_config(page_title="Advanced Emotion Detector", page_icon="😊", layout="wide")

st.markdown("""
<style>
body {
    background-color: #0f172a;
    color: white;
}
.uploadbox, .resultbox {
    background-color: #1e293b;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
}
.resultbox {
    margin-top: 20px;
}
.stButton>button {
    background-color: #3b82f6;
    color: white;
    border-radius: 8px;
}
.stButton>button:hover {
    background-color: #2563eb;
}
</style>
""", unsafe_allow_html=True)

st.title("😊 Advanced Emotion Detection Web App")
st.write("Upload an image or use your webcam to detect emotions, get advice, and enjoy related audio.")

# -------------------------
# SIDEBAR SETTINGS
# -------------------------
st.sidebar.title("Settings")
language = st.sidebar.selectbox("TTS Language", ["en", "es", "fr", "de"], index=0, help="Select language for audio advice.")
show_history = st.sidebar.checkbox("Show Detection History", value=True)
if st.sidebar.button("Reset History"):
    st.session_state.history = []

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# -------------------------
# MODEL LOADING
# -------------------------
@st.cache_resource
def load_emotion_model():
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_emotion_model()

@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(FACE_CASCADE_PATH)

face_cascade = load_face_cascade()

# -------------------------
# IMAGE PREPROCESSING AND FACE DETECTION
# -------------------------
def preprocess_and_detect_face(image):
    """Preprocess image and detect faces."""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None, "No face detected. Please upload an image with a clear face."
    
    # Use the first detected face
    x, y, w, h = faces[0]
    face_img = image.crop((x, y, x+w, y+h))
    face_img = ImageOps.grayscale(face_img)
    face_img = face_img.resize((48, 48))
    arr = np.array(face_img).astype("float32") / 255.0
    arr = arr.reshape(1, 48, 48, 1)
    return arr, None

# -------------------------
# AUDIO FUNCTION
# -------------------------
def play_audio_for_emotion(emotion, lang="en"):
    """Play audio for the detected emotion."""
    path = os.path.join("songs", SONG_FILES[emotion])
    if os.path.exists(path):
        st.audio(path)
    else:
        # Fallback to TTS
        tts = gTTS(ADVICE[emotion], lang=lang)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        st.audio(tmp.name)

# -------------------------
# SHARE RESULT FUNCTION
# -------------------------
def generate_share_link(emotion, confidence):
    """Generate a simple shareable link (base64 encoded for demo)."""
    data = f"Emotion: {emotion}, Confidence: {confidence:.2f}%"
    encoded = base64.b64encode(data.encode()).decode()
    return f"Share this: {st.get_option('server.headless')}?shared={encoded}"

# -------------------------
# MAIN APP
# -------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Input Options")
    input_method = st.radio("Choose Input Method", ["Upload Image", "Use Webcam"], index=0)
    
    if input_method == "Upload Image":
        uploaded = st.file_uploader("Upload an Image (JPG/PNG, max 5MB)", type=["jpg", "jpeg", "png"])
        if uploaded and uploaded.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error(f"File too large. Max size: {MAX_FILE_SIZE_MB}MB.")
            uploaded = None
        img = Image.open(uploaded) if uploaded else None
    else:
        img = st.camera_input("Take a Photo")
        if img:
            img = Image.open(img)
    
    if img:
        st.image(img, width=300, caption="Input Image")
        
        # Download option
        buf = BytesIO()
        img.save(buf, format="PNG")
        st.download_button("Download Image", buf.getvalue(), "processed_image.png", "image/png")

with col2:
    if img and model:
        with st.spinner("Detecting emotion..."):
            arr, error = preprocess_and_detect_face(img)
            if error:
                st.error(error)
            else:
                preds = model.predict(arr)
                label_idx = np.argmax(preds)
                emotion = EMOTION_LABELS[label_idx]
                confidence = preds[0][label_idx] * 100
                
                # Update history
                st.session_state.history.append({"emotion": emotion, "confidence": confidence})
                if len(st.session_state.history) > 10:  # Limit to last 10
                    st.session_state.history.pop(0)
                
                # Display result
                st.markdown(f"""
                <div class='resultbox'>
                    <h2>Detected Emotion: {emotion}</h2>
                    <p>Confidence: {confidence:.2f}%</p>
                    <p>{ADVICE[emotion]}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Play audio
                play_audio_for_emotion(emotion, language)
                
                # Share button
                share_link = generate_share_link(emotion, confidence)
                st.text_area("Shareable Link", share_link, height=50)
                st.button("Copy Link", on_click=lambda: st.write("Link copied! (Demo - implement clipboard in real app)"))

# -------------------------
# HISTORY SIDEBAR
# -------------------------
if show_history and st.session_state.history:
    st.sidebar.subheader("Recent Detections")
    for i, entry in enumerate(reversed(st.session_state.history)):
        st.sidebar.write(f"{i+1}. {entry['emotion']} ({entry['confidence']:.2f}%)")
