import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import joblib

@st.cache_data
def load_model_and_scaler():
    try:
        model = tf.keras.models.load_model('music_genre_cnn.h5', compile=False)
        
        scaler = joblib.load('scaler.joblib')
        genre_mapping = {
            0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop',
            5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'
        }
        
        return model, scaler, genre_mapping
    except Exception as e:
        st.error(f"An unexpected error occurred while loading model/scaler: {e}")
        st.stop()

def extract_features(audio_file, sample_rate=22050, n_mfcc=13, n_chroma=12):
    try:
        y, sr = librosa.load(audio_file, sr=sample_rate, duration=30)
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
        chroma_mean = np.mean(chroma, axis=1)
        
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_cent_mean = np.mean(spec_cent)
        
        spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spec_roll_mean = np.mean(spec_roll)
        
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        
        features = np.concatenate([
            mfccs_mean,
            chroma_mean,
            np.array([spec_cent_mean]),
            np.array([spec_roll_mean]),
            np.array([zcr_mean])
        ])
        return features

    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None

def predict_genre(audio_file):
    model, scaler, genre_mapping = load_model_and_scaler()

    features = extract_features(audio_file)

    if features is None:
        return "Error: Could not process audio file. Please try a different file."

    try:
        features_reshaped = features.reshape(1, -1)
        features_scaled = scaler.transform(features_reshaped)
    except Exception as e:
        st.error(f"Error during feature scaling: {e}")
        return "Error: Feature scaling failed."

    features_cnn = np.expand_dims(features_scaled, axis=-1)

    try:
        prediction_probs = model.predict(features_cnn)
    except Exception as e:
        st.error(f"Error during model prediction: {e}")
        return "Error: Model prediction failed."

    predicted_index = np.argmax(prediction_probs)
    predicted_genre = genre_mapping.get(predicted_index, "Unknown Genre")
    return predicted_genre


def main():
    st.title("🎵 Music Genre Classification App")
    st.write(
        "Welcome! This application uses a Convolutional Neural Network (CNN) to "
        "predict the genre of a music track."
    )
    st.write(
        "**Instructions:** Please upload a short audio file in `.wav` format to get started."
    )

    uploaded_file = st.file_uploader(
       "Drag and drop your audio file here",
        type=['wav']                          
    )
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        with st.spinner("Classifying your track... 🎶"):
            predicted_genre = predict_genre(uploaded_file)
        
        st.subheader("Prediction Result")
        st.markdown(f"## **{predicted_genre.capitalize()}**")


if __name__ == '__main__':
    main()
