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
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm



# Configuration
dataset_path = ".../datasets/RAVDESS_Interm"
output_path = ".../features/CoMISI/Ravdess/Noise"

# Initialize models
embedder = FaceNet()
ecapa_tdnn = SpeakerRecognition.from_hparams(source=".../pretrained_ecapa_tdnn")
detector = MTCNN()

def add_reverberation_and_noise(signal, sr=16000, snr_db_mean=30 , snr_db_std=1.5, sir_db_min=10, sir_db_max=25, room_dim=[8, 6, 3], absorption=0.6, mic_loc=[2, 3, 2], source_loc=[4, 3, 1.5]):
    
    
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




def preprocess_audio_librosa(audio_path, target_sample_rate=16000):
    # Load and resample the audio file
    signal, fs = librosa.load(audio_path, sr=target_sample_rate)
    # librosa ensures signal is mono by default, but let's ensure it's 1D
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    return signal, fs



def extract_audio_embedding(audio_path):
    try:
        signal, fs = preprocess_audio_librosa(audio_path, target_sample_rate=16000)
        if signal is not None:
            # Assuming add_reverberation_and_noise is adapted to work with numpy arrays directly
            processed_signal = add_reverberation_and_noise(signal, sr=fs)

            # Convert processed signal back to tensor, add batch dimension
            processed_signal_tensor = torch.tensor(processed_signal, dtype=torch.float32).unsqueeze(0)

            embeddings = ecapa_tdnn.encode_batch(processed_signal_tensor)

            if embeddings is not None and len(embeddings) > 0:
                return embeddings.flatten().cpu().numpy()  # Return embeddings as 1D numpy array
    except Exception as e:
        print(f"Error processing audio with librosa: {e}")
    return None  # Return None if preprocessing or encoding fails



def add_salt_pepper_noise(image, salt_pepper_ratio=0.02, amount_mean=0.06, amount_std=0.02):
    noisy_image = np.copy(image)
    amount = np.clip(np.random.normal(amount_mean, amount_std), 0, 1)
    num_salt = np.ceil(amount * image.size * salt_pepper_ratio)
    num_pepper = np.ceil(amount * image.size * (1.0 - salt_pepper_ratio))

    salt_coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1], :] = 1
    pepper_coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1], :] = 0
    return noisy_image


def apply_mild_blur(image, ksize_mean=19, ksize_std=5):
    """
    Applies mild Gaussian blur with Gaussian randomness in kernel size.
    """
    ksize = int(np.random.normal(ksize_mean, ksize_std))
    ksize = ksize if ksize % 2 == 1 else ksize + 1  # Ensure ksize is odd
    blurred_image = cv2.GaussianBlur(image, (ksize, ksize), 0)
    return blurred_image



def apply_color_shift(image, intensity_mean=30, intensity_std=15):
    """
    Shifts the color channels of the image with Gaussian randomness in intensity.
    """
    shifted_image = image.copy()
    for channel in range(image.shape[2]):
        intensity = int(np.random.normal(intensity_mean, intensity_std))
        shifted_image[:, :, channel] = np.clip(shifted_image[:, :, channel] + intensity, 0, 255)
    return shifted_image



def apply_intense_central_noise(image, intensity_mean=80, intensity_std=20, central_fraction=0.8):
    """
    Applies intense noise to a central region of the image with Gaussian randomness in intensity.
    """
    h, w = image.shape[:2]
    central_h, central_w = int(h * central_fraction), int(w * central_fraction)
    top, left = (h - central_h) // 2, (w - central_w) // 2

    intensity = int(np.random.normal(intensity_mean, intensity_std))
    central_region = image[top:top+central_h, left:left+central_w]
    noise = np.random.randint(-intensity, intensity, central_region.shape, dtype=np.int16)
    noisy_central_region = np.clip(central_region + noise, 0, 255).astype(np.uint8)

    image_with_noise = image.copy()
    image_with_noise[top:top+central_h, left:left+central_w] = noisy_central_region

    return image_with_noise



def random_rotation_scaling(image, angle_mean=15, angle_std=7, scale_mean=1.0, scale_std=0.05):
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


def adjusted_random_crop(image, crop_percentage_mean=0.05, crop_percentage_std=0.001):
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


    
def extract_visual_embedding(image_path):


    original_image = cv2.imread(image_path)  # Load the actual image from the path

    faces = detector.detect_faces(original_image)

    if faces:


        x, y, width, height = faces[0]['box']
        cropped_face = original_image[y:y+height, x:x+width]

        noisy_face = add_salt_pepper_noise(cropped_face)
        blurred_face = apply_mild_blur(noisy_face)
        color_shifted_face = apply_color_shift(blurred_face)  # Corrected to use `blurred_face`
        rotated_scaled_face = random_rotation_scaling(color_shifted_face)  # Using `color_shifted_face` for consistency
        face_with_central_noise = apply_intense_central_noise(rotated_scaled_face)
        final_face_image = adjusted_random_crop(face_with_central_noise)
        final_face_image_resized = cv2.resize(final_face_image, (160, 160))
        final_face_image_batch = np.expand_dims(final_face_image_resized, axis=0)
        
        embeddings = embedder.embeddings(final_face_image_batch)
        
        if embeddings is not None and embeddings.size > 0:
            return embeddings[0]

    return None






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