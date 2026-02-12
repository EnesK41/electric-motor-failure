import os
import scipy.io
import scipy.signal
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import config

class CWRUDataset(Dataset):
    def __init__(self, data_dir, feature_extractor):
        self.data_dir = data_dir
        self.feature_extractor = feature_extractor
        self.processed_data = [] # Cache processed tensors here

        file_list = [f for f in os.listdir(data_dir) if f.endswith(".mat")]
        
        print(f"Loading and processing dataset into RAM...")

        for file_name in tqdm(file_list, desc="Processing Files"):
            try:
                file_id = int(file_name.split('.')[0])
            except ValueError:
                continue

            label_text = self.get_label_from_id(file_id)

            try:
                file_path = os.path.join(self.data_dir, file_name)
                mat = scipy.io.loadmat(file_path)
                
                # Find the signal
                signal = None
                for key in mat.keys():
                    if 'DE_time' in key or 'FE_time' in key:
                        signal = mat[key].flatten()
                        break
                
                if signal is None: continue

                # Sliding Window
                chunk_size = config.CHUNK_SIZE
                step = config.STEP_SIZE
                
                for i in range(0, len(signal) - chunk_size, step):
                    chunk = signal[i : i + chunk_size]
                    
                    # 1. Resample to 16kHz
                    resampled = scipy.signal.resample(chunk, config.SAMPLE_RATE)
                    
                    # 2. Feature Extraction (Spectrogram)
                    inputs = self.feature_extractor(resampled, sampling_rate=config.SAMPLE_RATE, return_tensors="pt")
                    input_values = inputs['input_values'].squeeze(0)
                    
                    # Add to cache
                    self.processed_data.append((input_values, label_text))
                    
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

        print(f"Dataset ready! Total samples loaded: {len(self.processed_data)}")

    def get_label_from_id(self, fid):
        # 12k Drive End Bearing Fault Data
        if 97 <= fid <= 100: return "Normal Operation"
        if 105 <= fid <= 129:
            if fid in [105, 106, 107, 108]: return "Inner Race Fault 0.007 inch"
            if fid in [118, 119, 120, 121]: return "Ball Fault 0.007 inch"
            if fid in [130, 131, 132, 133]: return "Outer Race Fault 0.007 inch"
            return "Bearing Fault 0.007 inch"
        if 169 <= fid <= 200:
            if fid in [169, 170, 171, 172]: return "Inner Race Fault 0.014 inch"
            if fid in [185, 186, 187, 188]: return "Ball Fault 0.014 inch"
            if fid in [197, 198, 199, 200]: return "Outer Race Fault 0.014 inch"
            return "Bearing Fault 0.014 inch"
        if 209 <= fid <= 237:
            if fid in [209, 210, 211, 212]: return "Inner Race Fault 0.021 inch"
            if fid in [222, 223, 224, 225]: return "Ball Fault 0.021 inch"
            if fid in [234, 235, 236, 237]: return "Outer Race Fault 0.021 inch"
            return "Bearing Fault 0.021 inch"
        return "Unknown Fault"

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]