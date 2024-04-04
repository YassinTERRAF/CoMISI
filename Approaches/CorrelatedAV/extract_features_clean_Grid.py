import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mtcnn import MTCNN
from torchvision.transforms import ToTensor
import os
from tqdm import tqdm
import librosa
from torchvision import transforms
from PIL import Image
from keras_facenet import FaceNet
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

# Configuration
dataset_path = "lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/users/yassin.terraf/multimodal_speaker_recognition/datasets/GRID_Interm"
output_path = "lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/users/yassin.terraf/multimodal_speaker_recognition/features/Correlated/Grid/Clean"



# Initialize MTCNN face detector
detector = MTCNN()
embedder = FaceNet("20180408-102900")


def extract_mfcc_features(signal, sr=16000, n_mfcc=39,target_length=375):
    
    """
    Extracts MFCC features from an audio signal.

    Args:
        signal (np.ndarray): Audio signal array.
        sr (int): Sampling rate of the audio signal.
        n_mfcc (int): Number of MFCC features to extract.

    Returns:
        mfccs (np.ndarray): Extracted MFCC features as a 2D array.
    
    """
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)


    # Flatten the mfccs to a single vector
    mfcc_flattened = mfccs.flatten()

    # Determine the actual length of the flattened MFCCs
    actual_length = len(mfcc_flattened)

    # If the actual length is less than the target, pad with zeros
    if actual_length < target_length:
        return np.pad(mfcc_flattened, (0, target_length - actual_length), mode='constant')
    # If the actual length is more, truncate
    else:
   
        return mfcc_flattened[:target_length]





def extract_audio_embedding(audio_path, sr=16000, n_mfcc=39, target_length=375):
    """
    Extracts and processes audio embedding from an audio file.

    Args:
        audio_path (str): Path to the audio file.
        sr (int): Sampling rate for MFCC extraction.
        n_mfcc (int): Number of MFCC features to extract.
        target_length (int): Target length of the processed feature vector.

    Returns:
        aggregated_features (np.ndarray): Processed audio feature vector.
    """
    # Load the audio file
    signal, _ = librosa.load(audio_path, sr=sr)

    # Extract MFCC features
    mfccs = extract_mfcc_features(signal, sr, n_mfcc)


    return mfccs



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