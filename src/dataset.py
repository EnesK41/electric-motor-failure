import os
import scipy.io
import torch
import numpy as np
import scipy.signal
from torch.utils.data import Dataset

LABEL_MAPPING = {
    "97": "Normal", "98": "Normal",
    "105": "Inner Race Fault", "106": "Inner Race Fault", "107": "Inner Race Fault", "108": "Inner Race Fault",
    "118": "Ball Fault", "119": "Ball Fault", "120": "Ball Fault", "121": "Ball Fault",
    "130": "Outer Race Fault", "131": "Outer Race Fault", "132": "Outer Race Fault", "133": "Outer Race Fault"
}

class CWRUDataset(Dataset):
    def __init__(self, data_dir, feature_extractor):
        self.data_dir = data_dir
        self.feature_extractor = feature_extractor

        self.file_list = [f for f in os.listdir(data_dir) if f.endswith(".mat")]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_id = file_name.split('.')[0] #Labeling

        label_text = LABEL_MAPPING.get(file_id, "Unknown Condition")

        #Read the signal
        file_path = os.path.join(self.data_dir, file_name)
        mat = scipy.io.loadmat(file_path)

        signal = None
        for key in mat.keys():
            if 'DE_time' in key:
                signal = mat[key].flatten()
                break
        
        chunk = signal[:12000]  #First 1 second
        resampled = scipy.signal.resample(chunk, 16000) #turn it to ASM format

        inputs = self.feature_extractor(resampled, sampling_rate = 16000, return_tensors = "pt")
        
        return inputs['input_values'].squeeze(0), label_text

print("Dataset.py created") 
