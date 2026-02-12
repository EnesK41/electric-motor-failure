import os
import requests
import config

# List of files to download (CWRU Data IDs)
# Covers Normal, 0.007", 0.014", and 0.021" faults (12k Drive End)
FILES_TO_DOWNLOAD = [
    # Normal Baseline
    97, 98, 99, 100,
    # 0.007" Faults (Drive End)
    105, 106, 107, 108, 118, 119, 120, 121, 130, 131, 132, 133,
    # 0.014" Faults (Drive End)
    169, 170, 171, 172, 185, 186, 187, 188, 197, 198, 199, 200,
    # 0.021" Faults (Drive End)
    209, 210, 211, 212, 222, 223, 224, 225, 234, 235, 236, 237
]

BASE_URL = "https://engineering.case.edu/sites/default/files"

def download_file(file_id):
    filename = f"{file_id}.mat"
    url = f"{BASE_URL}/{filename}"
    save_path = os.path.join(config.RAW_DATA_DIR, filename)

    if os.path.exists(save_path):
        print(f"Skipping {filename} (Already exists)")
        return

    print(f"Downloading: {filename} ...")
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Successfully downloaded {filename}")
        else:
            print(f"Error: {filename} not found (Status: {response.status_code})")
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    if not os.path.exists(config.RAW_DATA_DIR):
        os.makedirs(config.RAW_DATA_DIR)
        print(f"Created directory: {config.RAW_DATA_DIR}")
    
    print(f"Starting CWRU Dataset Download...")
    print(f"Target Directory: {config.RAW_DATA_DIR}")
    print(f"Total Files: {len(FILES_TO_DOWNLOAD)}")
    
    for fid in FILES_TO_DOWNLOAD:
        download_file(fid)
    
    print("\nDownload process completed.")