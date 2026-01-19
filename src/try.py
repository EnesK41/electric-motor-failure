import os
import sys
import scipy.io

# Windows Tcl/Tk path fix
if os.name == 'nt':
    base_dir = sys.base_prefix
    tcl_dir = os.path.join(base_dir, 'tcl', 'tcl8.6')
    tk_dir = os.path.join(base_dir, 'tcl', 'tk8.6')
    if os.path.isdir(tcl_dir) and os.path.isdir(tk_dir):
        os.environ['TCL_LIBRARY'] = tcl_dir
        os.environ['TK_LIBRARY'] = tk_dir

import matplotlib.pyplot as plt
import numpy as np

# File paths
normal_path = 'data/raw/97.mat'
fault_path = 'data/raw/105.mat'

def plot_signal(file_path, title):
    try:
        # .mat dosyasını oku
        mat = scipy.io.loadmat(file_path)
        
        # Dosya içindeki değişkenleri bul (DE: Drive End, FE: Fan End)
        # Genelde 'X..._DE_time' formatında olur.
        for key in mat.keys():
            if 'DE_time' in key:
                signal_key = key
                break
        
        data = mat[signal_key]
        
        # İlk 1000 noktayı çizdir
        plt.figure(figsize=(10, 4))
        plt.plot(data[:1000])
        plt.title(f"{title} - Sinyal Anahtarı: {signal_key}")
        plt.ylabel("Titreşim Genliği")
        plt.xlabel("Zaman (Örnek)")
        plt.grid(True)
        plt.show()
        
        print(f"Successfully loaded {title}. Total data points: {len(data)}")
        
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Check 'data/raw' folder.")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Call the functions
plot_signal(normal_path, "Sağlam Motor (Normal)")
plot_signal(fault_path, "Arızalı Motor (Inner Race 0.007)")