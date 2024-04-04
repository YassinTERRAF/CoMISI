import os
import cv2
import numpy as np
import pandas as pd
import torch
from speechbrain.pretrained import SpeakerRecognition
import librosa
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm
from iresnet import IResNet
from iresnet import IResNet, IBasicBlock
from PIL import Image
from torchvision.transforms import ToPILImage
from torchvision.transforms import ToPILImage, Compose, Resize, ToTensor, Normalize
from torchvision.transforms.functional import to_tensor, to_pil_image
from mtcnn import MTCNN
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm


# Configuration
dataset_path = "lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/users/yassin.terraf/multimodal_speaker_recognition/datasets/GRID_Interm"
output_path = "lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/users/yassin.terraf/multimodal_speaker_recognition/features/CLEANSING/Gint/Grid/Clean"



# Initialize models
# Note: If not using FaceNet, you may remove or comment out its initialization
ecapa_tdnn = SpeakerRecognition.from_hparams(source="lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/users/yassin.terraf/multimodal_speaker_recognition/pretrained_ecapa_tdnn")
# Initialize and load the pretrained ResNet model correctly
pretrained_model_path = "lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/users/yassin.terraf/multimodal_speaker_recognition/Approaches/Cleansing/V-Glint.model"

resnet50_model = IResNet(block=IBasicBlock, model='res50', num_features=512)  # Adjust parameters as necessary
if torch.cuda.is_available():
    resnet50_model.load_state_dict(torch.load(pretrained_model_path))
else:
    resnet50_model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device('cpu')))
resnet50_model.eval()

# Initialize MTCNN detector
detector = MTCNN()


# Convert the cropped face tensor to a PIL Image
to_pil = ToPILImage()

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



def extract_visual_embedding(image_path):
    # Load the original image from the path
    original_image = cv2.imread(image_path)

    # Detect faces in the image
    faces = detector.detect_faces(original_image)

    if faces:
        # Assuming the first detected face is the one we are interested in
        x, y, width, height = faces[0]['box']
        # Crop the face from the original image
        cropped_face = original_image[y:y+height, x:x+width]

        # Convert the RGB image to PIL Image for compatibility with torchvision transforms
        cropped_face_pil = Image.fromarray(cropped_face)
 
        # Preprocess the cropped face
        preprocess = Compose([
            Resize((112, 112)),  # Resize the image to the expected input size of the ResNet model
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization parameters for the pretrained model
        ])

        face_tensor = preprocess(cropped_face_pil).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

        # Ensure the ResNet model is in evaluation mode
        resnet50_model.eval()
        with torch.no_grad():
            embedding = resnet50_model(face_tensor)

        return embedding.cpu().numpy().flatten()

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