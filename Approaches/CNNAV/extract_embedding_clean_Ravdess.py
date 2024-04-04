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
from mtcnn import MTCNN
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm



def preprocess_input(x, data_format=None, version=1):
    x_temp = np.copy(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if version == 1:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 93.5940
            x_temp[:, 1, :, :] -= 104.7624
            x_temp[:, 2, :, :] -= 129.1863
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 93.5940
            x_temp[..., 1] -= 104.7624
            x_temp[..., 2] -= 129.1863

    elif version == 2:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 91.4953
            x_temp[:, 1, :, :] -= 103.8827
            x_temp[:, 2, :, :] -= 131.0912
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 91.4953
            x_temp[..., 1] -= 103.8827
            x_temp[..., 2] -= 131.0912
    else:
        raise NotImplementedError

    return x_temp



#VGG
def init_vgg_model():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    model.load_weights('lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/users/yassin.terraf/multimodal_speaker_recognition/Approaches/CNNAV/vgg_face_weights.h5')
    return model


# Configuration
dataset_path = "lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/users/yassin.terraf/multimodal_speaker_recognition/datasets/RAVDESS_Interm"
output_path = "lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/users/yassin.terraf/multimodal_speaker_recognition/features/CNNAV/Ravdess/Clean"

# Initialize models
ecapa_tdnn = SpeakerRecognition.from_hparams(source="lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/users/yassin.terraf/multimodal_speaker_recognition/pretrained_ecapa_tdnn")
mtcnn = MTCNN()
vgg_model = init_vgg_model()






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



# Preprocessing for VGGFace
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img



def extract_visual_embedding(image_path, save_noisy_image=True):
    image = cv2.imread(image_path)
    # Detect faces using MTCNN
    results = mtcnn.detect_faces(image)
    if results:
        # Process the first detected face
        x, y, width, height = results[0]['box']
        face = image[y:y+height, x:x+width]
        face = cv2.resize(face, (224, 224))
        face_array = img_to_array(face)
        face_array = np.expand_dims(face_array, axis=0)
        face_array = preprocess_input(face_array, version=2)  # Adjust version if necessary
        # Extract embedding with VGG model
        embedding = vgg_model.predict(face_array)[0]
 
        return embedding
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