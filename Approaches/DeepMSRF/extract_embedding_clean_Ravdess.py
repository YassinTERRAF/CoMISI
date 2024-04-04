import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from mtcnn import MTCNN
import os 
from tqdm import tqdm
import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
from sklearn.model_selection import train_test_split



# Configuration
dataset_path = "lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/users/yassin.terraf/multimodal_speaker_recognition/datasets/RAVDESS_Interm"
output_path = "lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/users/yassin.terraf/multimodal_speaker_recognition/features/DeepMSRF/Ravdess/Clean"

mtcnn = MTCNN()

# Function to convert audio to a spectrogram
def audio_to_spectrogram(audio_path):
    signal, sr = librosa.load(audio_path, sr=16000)
    S = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB
    
# Custom VGG Model for Audio
class VGGAudio(nn.Module):
    def __init__(self):
        super(VGGAudio, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        # Modify the last layer for 4096-dimensional output (or your desired size)
        self.vgg.classifier[6] = nn.Linear(4096, 4096)

    def forward(self, x):
        return self.vgg(x)

# Initialize the audio model
vgg_audio_model = VGGAudio()
# Initialize the visual model similarly, if needed
vgg_visual_model = models.vgg16(pretrained=True)
vgg_visual_model.classifier[6] = nn.Linear(4096, 4096)  # Modify as necessary



def spectrogram_to_tensor(spectrogram):
    # Create a figure for plotting the spectrogram
    fig = Figure(figsize=(10, 4))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.axis('off')
    librosa.display.specshow(spectrogram, ax=ax, y_axis='mel', fmax=8000, x_axis='time')
    
    # Save the spectrogram to a buffer instead of a file
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    
    # Open the image directly from the buffer
    img = Image.open(buf).convert('RGB')
    
    # Transform the image to a tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # VGG's expected input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    tensor = transform(img)
    return tensor.unsqueeze(0)  # Add a batch dimension


# Function to extract audio embedding
def extract_audio_embedding(audio_path):
    spectrogram = audio_to_spectrogram(audio_path)
    tensor = spectrogram_to_tensor(spectrogram)
    with torch.no_grad():
        embedding = vgg_audio_model(tensor)
 
    return embedding.cpu().numpy().flatten()





# Function to extract visual embedding
def extract_visual_embedding(image_path):
    image = cv2.imread(image_path)
    

    results = mtcnn.detect_faces(image)
    if results:
        x, y, width, height = results[0]['box']
        cropped_face = image[y:y+height, x:x+width]
        cropped_face_pil = Image.fromarray(cropped_face)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = transform(cropped_face_pil).unsqueeze(0)

        with torch.no_grad():
            embedding = vgg_visual_model(tensor)


        return embedding.cpu().numpy().flatten()
    else:
        return None




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