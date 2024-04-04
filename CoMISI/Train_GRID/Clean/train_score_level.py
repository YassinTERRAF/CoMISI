import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from model_score_level import MultimodalNetworkWithScoreFusion  
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import os
import ast  # Add this import

# Constants
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY=0.9

class MultimodalDataset(Dataset):
    def __init__(self, csv_file):
        self.dataframe = pd.read_csv(csv_file)
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.dataframe['label'])

        # Use ast.literal_eval to safely evaluate the stringified list
        self.audio_features = self.dataframe['audio_embedding'].apply(lambda x: np.array(ast.literal_eval(x))).tolist()
        self.visual_features = self.dataframe['visual_embedding'].apply(lambda x: np.array(ast.literal_eval(x))).tolist()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        audio_feature = torch.tensor(self.audio_features[idx], dtype=torch.float32)
        visual_feature = torch.tensor(self.visual_features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return audio_feature, visual_feature, label



# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1

# Load data
train_dataset = MultimodalDataset(csv_file='lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/users/yassin.terraf/multimodal_speaker_recognition/features/CoMISI/Grid/Clean/train_features.csv')
val_dataset = MultimodalDataset(csv_file='lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/users/yassin.terraf/multimodal_speaker_recognition/features/CoMISI/Grid/Clean/val_features.csv')
test_dataset = MultimodalDataset(csv_file='lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/users/yassin.terraf/multimodal_speaker_recognition/features/CoMISI/Grid/Clean/test_features.csv')

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)




# Define the training and evaluation process
def train_and_evaluate(model, train_loader, val_loader, test_loader, patience=5):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    
    best_val_loss = float('inf')
    best_model_path = 'lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/users/yassin.terraf/multimodal_speaker_recognition/CoMISI/weights/best_model_score_Grid_clean.pth'
    epochs_no_improve = 0  # Initialize the counter for epochs with no improvement

    for epoch in range(EPOCHS):
        model.train()
        for audio_feature, visual_feature, label in train_loader:
            optimizer.zero_grad()
            outputs = model(audio_feature, visual_feature)  # Assuming model takes audio and visual features
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for audio_feature, visual_feature, label in val_loader:
                outputs = model(audio_feature, visual_feature)
                loss = criterion(outputs, label)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch+1}: New best model saved with loss {best_val_loss}")
            epochs_no_improve = 0  # Reset counter
        else:
            epochs_no_improve += 1
            print(f"Epoch {epoch+1}: No improvement in validation loss. Current streak: {epochs_no_improve} epochs.")

        # Early stopping check
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    # Load the best model for evaluation
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Test loop
    test_preds, test_labels = [], []
    with torch.no_grad():
        for audio_feature, visual_feature, label in test_loader:
            outputs = model(audio_feature, visual_feature)
            preds = torch.argmax(outputs, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(label.cpu().numpy())

    accuracy, precision, recall, f1 = calculate_metrics(test_labels, test_preds)
    return accuracy, precision, recall, f1

if __name__ == "__main__":
    NUM_RUNS = 10
    all_results = []  # Store results of all runs

    for run in range(NUM_RUNS):
        print(f"\nStarting run {run+1}/{NUM_RUNS}")

        # Re-initialize the model for each run
        model = MultimodalNetworkWithScoreFusion(dropout_rate=0.5)  # Ensure MultimodalNetwork is correctly defined/imported

        # Train and evaluate the model
        test_accuracy, test_precision, test_recall, test_f1 = train_and_evaluate(model, train_loader, val_loader, test_loader)

        # Store results of this run
        all_results.append((test_accuracy, test_precision, test_recall, test_f1))

        # Print results for this run
        print(f"Run {run+1}: Test Accuracy: {test_accuracy}, Test Precision: {test_precision}, Test Recall: {test_recall}, Test F1: {test_f1}")

    # Calculate average metrics over all runs
    avg_accuracy = np.mean([result[0] for result in all_results])
    avg_precision = np.mean([result[1] for result in all_results])
    avg_recall = np.mean([result[2] for result in all_results])
    avg_f1 = np.mean([result[3] for result in all_results])

    # Write results to a file for the single run
    results_file_path = 'lustre/rsc_mgt_wi-st-sccs-lj7uaansp4q/users/yassin.terraf/multimodal_speaker_recognition/Results/CoMISI/GRID/clean/score_level_Results.txt'  # Adjust the path as necessary
    with open(results_file_path, 'w') as file:
        for run, (accuracy, precision, recall, f1) in enumerate(all_results, start=1):
            file.write(f'Run {run}: Test Accuracy: {accuracy}, Test Precision: {precision}, Test Recall: {recall}, Test F1: {f1}\n')
        file.write(f'\nAverage over {NUM_RUNS} runs: Test Accuracy: {avg_accuracy}, Test Precision: {avg_precision}, Test Recall: {avg_recall}, Test F1: {avg_f1}\n')

    print(f"\nResults have been saved to {results_file_path}")