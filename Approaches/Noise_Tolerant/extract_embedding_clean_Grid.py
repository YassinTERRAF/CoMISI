import os
import cv2
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F  # Ensure this import is included
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras_facenet import FaceNet
from speechbrain.pretrained import SpeakerRecognition
from skimage.io import imsave
import pyroomacoustics as pra
from mtcnn import MTCNN
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm


# Configuration
dataset_path = "lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/users/yassin.terraf/multimodal_speaker_recognition/datasets/GRID_Interm"
output_path = "lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/users/yassin.terraf/multimodal_speaker_recognition/features/Noise_Tolerant/Grid/Clean"

# Initialize MTCNN face detector
detector = MTCNN()
embedder = FaceNet("20180408-102900")





class CustomAudioModel(nn.Module):
    def __init__(self):
        super(CustomAudioModel, self).__init__()
        # Define the convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=40, out_channels=1000, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=1000, out_channels=1000, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=1000, out_channels=1000, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=1000, out_channels=1500, kernel_size=1, stride=1),
            nn.ReLU()
        )
        # Fully connected layers
        self.fc1 = nn.Linear(1500, 1500)
        self.fc2 = nn.Linear(1500, 600)
    
    def forward(self, x):
        x = self.conv_layers(x)
        # Emulating Global Average Pooling (Statistics Pooling)
        x = torch.mean(x, dim=2)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

custom_audio_model = CustomAudioModel()

def compute_mfcc(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, norm='ortho')
    mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True))  # Zero-mean normalization
    mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    return mfcc_tensor


def extract_audio_embedding(audio_path):
    mfcc_features = compute_mfcc(audio_path)
    with torch.no_grad():
        audio_embedding = custom_audio_model(mfcc_features)
    audio_embedding = audio_embedding.cpu().numpy().flatten()
    return audio_embedding


def extract_visual_embedding(image_path, save_noisy_image=False):
    # Load the original image from the path
    original_image = cv2.imread(image_path)

    # Detect faces in the image
    faces = detector.detect_faces(original_image)

    if faces:
        # Assuming the first detected face is the one we are interested in
        x, y, width, height = faces[0]['box']
        # Crop the face from the original image
        cropped_face = original_image[y:y+height, x:x+width]

        # Resize the cropped face image to 160x160 as required by FaceNet
        final_face_image_resized = cv2.resize(cropped_face, (160, 160))

        # Prepare the image for embedding extraction
        final_face_image_batch = np.expand_dims(final_face_image_resized, axis=0)

        # Extract embeddings for the pre-cropped and resized face image
        embeddings = embedder.embeddings(final_face_image_batch)

        # Check and return the embeddings
        if embeddings is not None and embeddings.size > 0:
            return embeddings[0]  # Assuming you're processing one image at a time

    return None  # Return None if no faces are detected or if embeddings couldn't be extracted







def process_file(audio_path, image_path, speaker_label):
    """
    Process a single file pair (audio and image) and return the feature dict.
    """
    # Ensure both audio and image files exist
    if os.path.exists(audio_path) and os.path.exists(image_path):

        audio_embedding = extract_audio_embedding(audio_path)
        visual_embedding = extract_visual_embedding(image_path)

        if visual_embedding is not None:
            return {
                "audio_embedding": list(audio_embedding),
                "visual_embedding": list(visual_embedding),
                "label": speaker_label
            }
    return None


def process_files_parallel(dataset_path):
    all_features = []
    speaker_dirs = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    with ThreadPoolExecutor() as executor:
        futures = []

        for speaker_dir in speaker_dirs:
            files = os.listdir(speaker_dir)
            for file in files:
                if file.endswith('.wav'):
                    base_filename = file[:-4]  # Remove .wav extension
                    audio_path = os.path.join(speaker_dir, file)
                    image_path = os.path.join(speaker_dir, base_filename + '.jpg')
                    speaker_label = speaker_dir.split(os.sep)[-1]  # Extract speaker label from the folder name

                    futures.append(executor.submit(process_file, audio_path, image_path, speaker_label))

        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            result = future.result()
            if result:
                all_features.append(result)

    return pd.DataFrame(all_features)


# Process the dataset to extract embeddings
features_df = process_files_parallel(dataset_path)



train_df, test_df = train_test_split(features_df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

train_df.to_csv(os.path.join(output_path, 'train_features.csv'), index=False)
val_df.to_csv(os.path.join(output_path, 'val_features.csv'), index=False)
test_df.to_csv(os.path.join(output_path, 'test_features.csv'), index=False)