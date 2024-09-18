import cv2
import numpy as np
import os
import pandas as pd
from keras_facenet import FaceNet
from speechbrain.pretrained import SpeakerRecognition
import torchaudio
import librosa
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from skimage.io import imsave
import pyroomacoustics as pra
from mtcnn import MTCNN
import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm


# Configuration
dataset_path = ".../datasets/GRID_Interm"
output_path = ".../features/CoMISI/Grid/Clean"

# Initialize models
embedder = FaceNet()
ecapa_tdnn = SpeakerRecognition.from_hparams(source=".../pretrained_ecapa_tdnn")
detector = MTCNN()

def preprocess_audio_librosa(audio_path, target_sample_rate=16000):
    # Load the audio file with librosa, resampling to the target rate in one step
    signal, fs = librosa.load(audio_path, sr=target_sample_rate)
    # Ensure the signal is mono (1D) by averaging the channels if it's stereo (2D)
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    return signal, fs

def extract_audio_embedding(audio_path):
    try:
        signal, fs = preprocess_audio_librosa(audio_path, target_sample_rate=16000)
        # Ensure the signal has a batch dimension if needed by the model
        signal = np.expand_dims(signal, axis=0)
        # Convert the signal to a tensor for compatibility with the model, if necessary
        signal_tensor = torch.from_numpy(signal).float()
        if signal is not None:
            # Assuming ecapa_tdnn can process numpy arrays directly, or use signal_tensor if it requires torch.Tensor
            embeddings = ecapa_tdnn.encode_batch(signal_tensor)
            if embeddings is not None and len(embeddings) > 0:
                return embeddings.flatten().cpu().numpy()  # Return embeddings as 1D numpy array
    
    except Exception as e:
        print(f"Error processing audio with librosa: {e}")
    return None  # Return None if preprocessing or encoding fails



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

        if audio_embedding is not None and visual_embedding is not None:
            return {
                "audio_embedding": list(audio_embedding),
                "visual_embedding": list(visual_embedding),
                "label": speaker_label
            }
    return None



def process_files_parallel(dataset_path):
    all_features = []
    speaker_dirs = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    # Determine the maximum number of CPUs available
    max_workers = os.cpu_count()


    with ThreadPoolExecutor(max_workers=max_workers) as executor:
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

# Split the DataFrame into train, validation, and test sets
train_df, test_df = train_test_split(features_df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)  # Adjust to achieve 15% for both val and test



# Save the splits to CSV files
train_df.to_csv(os.path.join(output_path, 'train_features.csv'), index=False)
val_df.to_csv(os.path.join(output_path, 'val_features.csv'), index=False)
test_df.to_csv(os.path.join(output_path, 'test_features.csv'), index=False)