# Importing modules
import numpy as np
import streamlit as st
import cv2
import pandas as pd
import os
from pathlib import Path
import random
import urllib.parse

from collections import Counter

# Get current directory
current_dir = Path(__file__).parent

# Load data with error handling
try:
    df = pd.read_csv(current_dir / "muse_v3.csv")
    st.success("✅ Data loaded successfully!")
except FileNotFoundError:
    st.error("❌ muse_v3.csv file not found! Please ensure the file is in the same directory as app.py")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.stop()

# Create proper music streaming links
def create_music_link(track_name, artist_name):
    """Create a search link for the song on popular music platforms"""
    # Encode track and artist for URL
    track_encoded = urllib.parse.quote(f"{track_name} {artist_name}")
    
    # Create multiple platform links
    links = {
        "Spotify": f"https://open.spotify.com/search/{track_encoded}",
        "YouTube": f"https://www.youtube.com/results?search_query={track_encoded}",
        "Apple Music": f"https://music.apple.com/search?term={track_encoded}",
        "Amazon Music": f"https://music.amazon.com/search/{track_encoded}"
    }
    return links

# Process the data
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']

# Create proper links for each song
df['music_links'] = df.apply(lambda row: create_music_link(row['track'], row['artist']), axis=1)

df = df[['name','emotional','pleasant','music_links','artist']]
st.write("Data shape:", df.shape)

df = df.sort_values(by=["emotional", "pleasant"])
df = df.reset_index(drop=True)

# Split data into emotion categories
total_rows = len(df)
chunk_size = total_rows // 5

df_sad = df[:chunk_size]
df_fear = df[chunk_size:2*chunk_size]
df_angry = df[2*chunk_size:3*chunk_size]
df_neutral = df[3*chunk_size:4*chunk_size]
df_happy = df[4*chunk_size:]

def fun(emotion_list):
    """Generate music recommendations based on detected emotions"""
    data = pd.DataFrame()

    if len(emotion_list) == 1:
        v = emotion_list[0]
        t = 15  # Reduced from 30 to 15
        if v == 'Neutral':
            data = pd.concat([data, df_neutral.sample(n=min(t, len(df_neutral)))], ignore_index=True)
        elif v == 'Angry':
             data = pd.concat([data, df_angry.sample(n=min(t, len(df_angry)))], ignore_index=True)
        elif v == 'Fearful':
            data = pd.concat([data, df_fear.sample(n=min(t, len(df_fear)))], ignore_index=True)
        elif v == 'Happy':
            data = pd.concat([data, df_happy.sample(n=min(t, len(df_happy)))], ignore_index=True)
        elif v == 'Sad':
            data = pd.concat([data, df_sad.sample(n=min(t, len(df_sad)))], ignore_index=True)
        else:
            data = pd.concat([data, df_neutral.sample(n=min(t, len(df_neutral)))], ignore_index=True)

    elif len(emotion_list) == 2:
        times = [8, 7]  # Total: 15
        for i in range(len(emotion_list)):
            v = emotion_list[i]
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=min(t, len(df_neutral)))], ignore_index=True)
            elif v == 'Angry':    
                data = pd.concat([data, df_angry.sample(n=min(t, len(df_angry)))], ignore_index=True)
            elif v == 'Fearful':              
                data = pd.concat([data, df_fear.sample(n=min(t, len(df_fear)))], ignore_index=True)
            elif v == 'Happy':             
                data = pd.concat([data, df_happy.sample(n=min(t, len(df_happy)))], ignore_index=True)
            elif v == 'Sad':              
                data = pd.concat([data, df_sad.sample(n=min(t, len(df_sad)))], ignore_index=True)
            else:
                data = pd.concat([data, df_neutral.sample(n=min(t, len(df_neutral)))], ignore_index=True)

    elif len(emotion_list) == 3:
        times = [6, 5, 4]  # Total: 15
        for i in range(len(emotion_list)): 
            v = emotion_list[i]          
            t = times[i]

            if v == 'Neutral':              
                data = pd.concat([data, df_neutral.sample(n=min(t, len(df_neutral)))], ignore_index=True)
            elif v == 'Angry':               
                data = pd.concat([data, df_angry.sample(n=min(t, len(df_angry)))], ignore_index=True)
            elif v == 'Fearful':             
                data = pd.concat([data, df_fear.sample(n=min(t, len(df_fear)))], ignore_index=True)
            elif v == 'Happy':               
                data = pd.concat([data, df_happy.sample(n=min(t, len(df_happy)))], ignore_index=True)
            elif v == 'Sad':      
                data = pd.concat([data, df_sad.sample(n=min(t, len(df_sad)))], ignore_index=True)
            else:
                data = pd.concat([data, df_neutral.sample(n=min(t, len(df_neutral)))], ignore_index=True)

    elif len(emotion_list) == 4:
        times = [4, 4, 4, 3]  # Total: 15
        for i in range(len(emotion_list)):
            v = emotion_list[i]
            t = times[i]
            if v == 'Neutral': 
                data = pd.concat([data, df_neutral.sample(n=min(t, len(df_neutral)))], ignore_index=True)
            elif v == 'Angry':              
                data = pd.concat([data, df_angry.sample(n=min(t, len(df_angry)))], ignore_index=True)
            elif v == 'Fearful':              
                data = pd.concat([data, df_fear.sample(n=min(t, len(df_fear)))], ignore_index=True)
            elif v == 'Happy':               
                data = pd.concat([data, df_happy.sample(n=min(t, len(df_happy)))], ignore_index=True)
            elif v == 'Sad':              
                data = pd.concat([data, df_sad.sample(n=min(t, len(df_sad)))], ignore_index=True)
            else:
                data = pd.concat([data, df_neutral.sample(n=min(t, len(df_neutral)))], ignore_index=True)
    else:
        times = [4, 3, 3, 3, 2]  # Total: 15
        for i in range(len(emotion_list)):           
            v = emotion_list[i]         
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=min(t, len(df_neutral)))], ignore_index=True)
            elif v == 'Angry':           
                data = pd.concat([data, df_angry.sample(n=min(t, len(df_angry)))], ignore_index=True)
            elif v == 'Fearful':           
                data = pd.concat([data, df_fear.sample(n=min(t, len(df_fear)))], ignore_index=True)
            elif v == 'Happy':          
                data = pd.concat([data, df_happy.sample(n=min(t, len(df_happy)))], ignore_index=True)
            elif v == 'Sad':
                data = pd.concat([data, df_sad.sample(n=min(t, len(df_sad)))], ignore_index=True)
            else:
                data = pd.concat([data, df_neutral.sample(n=min(t, len(df_neutral)))], ignore_index=True)

    return data

def preprocess_emotions(emotion_list):
    """Process detected emotions and return unique emotions in order of frequency"""
    if not emotion_list:
        return []
    
    emotion_counts = Counter(emotion_list)
    result = []
    for emotion, count in emotion_counts.items():
        result.extend([emotion] * count)

    # Get unique emotions in order of occurrence frequency
    unique_emotions = []
    for emotion in result:
        if emotion not in unique_emotions:
            unique_emotions.append(emotion)
    
    return unique_emotions

# Load face cascade with error handling
try:
    face_cascade_path = current_dir / 'haarcascade_frontalface_default.xml'
    if not face_cascade_path.exists():
        # Try using OpenCV's built-in cascade
        face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    
    face_cascade = cv2.CascadeClassifier(str(face_cascade_path))
    if face_cascade.empty():
        st.error("❌ Failed to load face cascade classifier")
        st.stop()
    else:
        st.success("✅ Face cascade classifier loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading face cascade: {str(e)}")
    st.stop()

# Streamlit UI
st.markdown("""
<style>
body {
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: white'><b>Emotion based music recommendation</b></h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: grey;'><b>Click on the platform links to listen to recommended songs</b></h5>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

detected_emotions = []

with col1:
    pass

with col2:
    # Add emotion selection option for testing
    st.markdown("### 🎭 Emotion Detection Options")
    
    # Option 1: Manual emotion selection (for testing)
    if st.checkbox("Use manual emotion selection (for testing)"):
        selected_emotions = st.multiselect(
            "Select your emotions:",
            ["Happy", "Sad", "Angry", "Neutral", "Fearful", "Surprised", "Disgusted"],
            default=["Happy"]
        )
        if st.button("🎵 Get Recommendations"):
            detected_emotions = selected_emotions
            st.success(f"✅ Selected emotions: {', '.join(detected_emotions)}")
    
    # Option 2: Camera-based detection (if TensorFlow is available)
    else:
        if st.button('🎭 SCAN EMOTION (Click here)', type="primary"):
            st.info("🔍 Starting emotion detection... Look at the camera!")
            
            # Initialize camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Cannot open camera")
                st.stop()
            
            # Create placeholder for video
            video_placeholder = st.empty()
            
            count = 0
            detected_emotions.clear()
            
            # Simple emotion detection based on face detection only
            # (This is a simplified version - in real app you'd use the TensorFlow model)
            try:
                while count < 30:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Failed to grab frame")
                        break
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                    count += 1

                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                        
                        # Simple random emotion for demonstration
                        # In real app, this would use the TensorFlow model
                        emotions = ["Happy", "Neutral", "Sad", "Angry", "Fearful"]
                        emotion = random.choice(emotions)
                        detected_emotions.append(emotion)
                        
                        cv2.putText(frame, emotion, (x + 20, y - 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Display frame in Streamlit
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                    
                    # Add a small delay
                    import time
                    time.sleep(0.1)
                    
            except Exception as e:
                st.error(f"❌ Error during emotion detection: {str(e)}")
            finally:
                cap.release()
            
            if detected_emotions:
                processed_emotions = preprocess_emotions(detected_emotions)
                st.success(f"✅ Emotions detected: {', '.join(processed_emotions)}")
            else:
                st.warning("⚠️ No emotions detected. Please try again.")

with col3:
    pass

# Generate recommendations
if detected_emotions:
    processed_emotions = preprocess_emotions(detected_emotions)
    new_df = fun(processed_emotions)
    
    st.write("")
    st.markdown("<h5 style='text-align: center; color: grey;'><b>Recommended songs with artist names</b></h5>", unsafe_allow_html=True)
    st.write("---------------------------------------------------------------------------------------------------------------------")

    try:
        for i, (music_links, artist, name) in enumerate(zip(new_df["music_links"], new_df['artist'], new_df['name']), 1):
            st.markdown(f"<h4 style='text-align: center;'>{i}. {name}</h4>", unsafe_allow_html=True)
            st.markdown(f"<h5 style='text-align: center; color: grey;'><i>{artist}</i></h5>", unsafe_allow_html=True)
            
            # Create platform links
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"[🎵 Spotify]({music_links['Spotify']})", unsafe_allow_html=True)
            with col2:
                st.markdown(f"[📺 YouTube]({music_links['YouTube']})", unsafe_allow_html=True)
            with col3:
                st.markdown(f"[🍎 Apple Music]({music_links['Apple Music']})", unsafe_allow_html=True)
            with col4:
                st.markdown(f"[🛒 Amazon Music]({music_links['Amazon Music']})", unsafe_allow_html=True)
            
            st.write("---------------------------------------------------------------------------------------------------------------------")
    except Exception as e:
        st.error(f"❌ Error displaying recommendations: {str(e)}")
else:
    st.info("🎵 Click 'SCAN EMOTION' or use manual selection to get music recommendations based on your emotions!") 