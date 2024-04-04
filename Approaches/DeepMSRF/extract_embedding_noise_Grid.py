import os
import cv2
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation
from mtcnn import MTCNN
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
import librosa
from speechbrain.pretrained import SpeakerRecognition
from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K
import pyroomacoustics as pra
from keras_facenet import FaceNet
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm
import io


# Configuration
dataset_path = "lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/users/yassin.terraf/multimodal_speaker_recognition/datasets/GRID_Interm"
output_path = "lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/users/yassin.terraf/multimodal_speaker_recognition/features/DeepMSRF/Grid/Noise"





mtcnn = MTCNN()

# Function to generate spectrogram from a signal
def audio_to_spectrogram_from_signal(signal, sr=16000):
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



def add_reverberation_and_noise(signal, sr=16000, snr_db_mean=30 , snr_db_std=1.5, sir_db_min=5, sir_db_max=25, room_dim=[8, 6, 3], absorption=0.5, mic_loc=[2, 3, 2], source_loc=[4, 3, 1.5]):
    
    
    room = pra.ShoeBox(room_dim, absorption=absorption, fs=sr, max_order=15)
    room.add_source(source_loc, signal=signal)
    room.add_microphone_array(pra.MicrophoneArray(np.array([mic_loc]).T, room.fs))
    room.simulate()

    mic_signal = room.mic_array.signals[0, :]
    snr_db = np.random.normal(snr_db_mean, snr_db_std)
    sir_db = np.random.uniform(sir_db_min, sir_db_max)  # Uniform distribution for SIR
    signal_power = np.mean(mic_signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    interference_power = signal_power / (10 ** (sir_db / 10))

    noise = np.sqrt(noise_power) * np.random.randn(len(mic_signal))
    interference = np.sqrt(interference_power) * np.random.randn(len(mic_signal))
    noisy_signal = mic_signal + noise + interference
    return noisy_signal





# Update the audio processing to add reverberation and noise before converting to a spectrogram
def extract_audio_embedding(audio_path):
    signal, sr = librosa.load(audio_path, sr=16000)
    # Add reverberation and noise
    noisy_signal = add_reverberation_and_noise(signal, sr=sr)
    # Generate the spectrogram from the noisy signal
    spectrogram = audio_to_spectrogram_from_signal(noisy_signal, sr)
    tensor = spectrogram_to_tensor(spectrogram)
    with torch.no_grad():
        embedding = vgg_audio_model(tensor)
    return embedding.cpu().numpy().flatten()








def add_salt_pepper_noise(image, salt_pepper_ratio=0.02, amount_mean=0.01, amount_std=0.008):
    noisy_image = np.copy(image)
    amount = np.clip(np.random.normal(amount_mean, amount_std), 0, 1)
    num_salt = np.ceil(amount * image.size * salt_pepper_ratio)
    num_pepper = np.ceil(amount * image.size * (1.0 - salt_pepper_ratio))

    salt_coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1], :] = 1
    pepper_coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1], :] = 0
    return noisy_image


def apply_mild_blur(image, ksize_mean=13, ksize_std=3):
    """
    Applies mild Gaussian blur with Gaussian randomness in kernel size.
    Ensures kernel size (ksize) is a positive odd number.
    """
    ksize = int(abs(np.random.normal(ksize_mean, ksize_std)))
    ksize = max(1, ksize)  # Ensure ksize is at least 1
    ksize = ksize if ksize % 2 == 1 else ksize + 1  # Ensure ksize is odd
    blurred_image = cv2.GaussianBlur(image, (ksize, ksize), 0)
    return blurred_image


def apply_color_shift(image, intensity_mean=12, intensity_std=3):
    """
    Shifts the color channels of the image with Gaussian randomness in intensity.
    """
    shifted_image = image.copy()
    for channel in range(image.shape[2]):
        intensity = int(np.random.normal(intensity_mean, intensity_std))
        shifted_image[:, :, channel] = np.clip(shifted_image[:, :, channel] + intensity, 0, 255)
    return shifted_image



def apply_intense_central_noise(image, intensity_mean=50, intensity_std=2, central_fraction=0.9):
    """
    Applies intense noise to a central region of the image with Gaussian randomness in intensity.
    """
    h, w = image.shape[:2]
    central_h, central_w = int(h * central_fraction), int(w * central_fraction)
    top, left = (h - central_h) // 2, (w - central_w) // 2

    # Ensure intensity is at least 1 to avoid low >= high error in np.random.randint
    intensity = max(1, abs(int(np.random.normal(intensity_mean, intensity_std))))
    
    central_region = image[top:top+central_h, left:left+central_w]
    noise = np.random.randint(-intensity, intensity, central_region.shape, dtype=np.int16)
    noisy_central_region = np.clip(central_region + noise, 0, 255).astype(np.uint8)

    image_with_noise = image.copy()
    image_with_noise[top:top+central_h, left:left+central_w] = noisy_central_region

    return image_with_noise



def random_rotation_scaling(image, angle_mean=5, angle_std=2, scale_mean=1.0, scale_std=0.05):
    """
    Randomly rotates and scales an image with Gaussian randomness in angle and scale.
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    angle = np.random.normal(angle_mean, angle_std)
    scale = np.random.normal(scale_mean, scale_std)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_scaled_image = cv2.warpAffine(image, M, (w, h))

    return rotated_scaled_image



def adjusted_random_crop(image, crop_percentage_mean=0.04, crop_percentage_std=0.001):
    """
    Randomly crops a smaller region from the edges of the image to simulate off-center subjects
    or framing variations, with Gaussian randomness in the crop percentage. The cropped image
    is then resized back to the original dimensions to simulate zooming.
    
    Parameters:
    - image: Input image.
    - crop_percentage_mean: Mean percentage of the image to crop from each side.
    - crop_percentage_std: Standard deviation of the crop percentage.
    
    Returns:
    - Cropped and resized image.
    """
    h, w = image.shape[:2]
    # Ensure crop percentage is within 0 to 0.5 range to prevent negative dimensions
    crop_percentage = np.clip(np.random.normal(crop_percentage_mean, crop_percentage_std), 0, 0.5)
    
    crop_h, crop_w = int(h * crop_percentage), int(w * crop_percentage)
    # Calculate the coordinates for the crop to maintain the center
    start_y, end_y = crop_h, h - crop_h
    start_x, end_x = crop_w, w - crop_w
    
    # Crop the image
    crop_image = image[start_y:end_y, start_x:end_x]
    
    # Resize the cropped image back to original dimensions
    resized_image = cv2.resize(crop_image, (w, h))
    
    return resized_image





def extract_visual_embedding(image_path, save_noisy_image=True):
    image = cv2.imread(image_path)
    # Detect faces using MTCNN
    detector = MTCNN()
    results = detector.detect_faces(image)
    if results:
        # Process the first detected face
        x, y, width, height = results[0]['box']
        face = image[y:y+height, x:x+width]

        # Apply the sequence of distortions to the cropped face
        noisy_face = add_salt_pepper_noise(face)
        blurred_face = apply_mild_blur(noisy_face)
        color_shifted_face = apply_color_shift(blurred_face)
        rotated_scaled_face = random_rotation_scaling(color_shifted_face)
        face_with_central_noise = apply_intense_central_noise(rotated_scaled_face)
        transformed_face = adjusted_random_crop(face_with_central_noise)

        # Convert the processed face image to PIL Image for torchvision transforms
        transformed_face_pil = Image.fromarray(transformed_face)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = transform(transformed_face_pil).unsqueeze(0)

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