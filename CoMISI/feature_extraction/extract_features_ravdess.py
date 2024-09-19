import os
import numpy as np
import cv2
import torch
import librosa
from keras_facenet import FaceNet
from speechbrain.pretrained import SpeakerRecognition
from mtcnn import MTCNN

# Initialize models
embedder = FaceNet()
ecapa_tdnn = SpeakerRecognition.from_hparams(source=".../pretrained_ecapa_tdnn")
detector = MTCNN()

def preprocess_audio_librosa(audio_path, target_sample_rate=16000):
    """
    Load and resample the audio file to the target sample rate.
    """
    signal, fs = librosa.load(audio_path, sr=target_sample_rate)
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    return signal, fs

def extract_audio_embedding(audio_path):
    """
    Extract audio embeddings using ECAPA-TDNN from SpeechBrain.
    """
    try:
        signal, fs = preprocess_audio_librosa(audio_path, target_sample_rate=16000)
        signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        embeddings = ecapa_tdnn.encode_batch(signal_tensor)

        if embeddings is not None and len(embeddings) > 0:
            return embeddings.flatten().cpu().numpy()  # Return embeddings as 1D numpy array
    except Exception as e:
        print(f"Error processing audio: {e}")
    return None

def extract_visual_embedding(image_path):
    """
    Extract visual embeddings using FaceNet.
    """
    original_image = cv2.imread(image_path)
    faces = detector.detect_faces(original_image)

    if faces:
        x, y, width, height = faces[0]['box']
        cropped_face = original_image[y:y+height, x:x+width]
        final_face_image_resized = cv2.resize(cropped_face, (160, 160))
        final_face_image_batch = np.expand_dims(final_face_image_resized, axis=0)
        embeddings = embedder.embeddings(final_face_image_batch)

        if embeddings is not None and embeddings.size > 0:
            return embeddings[0]
    return None

def process_file(audio_path, image_path, speaker_label):
    """
    Process a single file pair (audio and image) and return the feature dictionary.
    """
    if os.path.exists(audio_path) and os.path.exists(image_path):
        audio_embedding = extract_audio_embedding(audio_path)
        visual_embedding = extract_visual_embedding(image_path)

        if audio_embedding is not None and visual_embedding is not None:
            return {
                "audio_embedding": list(audio_embedding),
                "visual_embedding": list(visual_embedding),
                "label": speaker_label
            }
    return None
