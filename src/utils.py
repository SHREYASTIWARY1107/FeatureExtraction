# src/utils.py

import re
import os
import requests
import pandas as pd
import multiprocessing
import time
from tqdm import tqdm
import numpy as np
from pathlib import Path
from PIL import Image
from .constants import allowed_units  # Use relative import
from functools import partial

def preprocess_text(text):
    """Preprocess the given text by removing punctuation, converting to lowercase, and removing extra whitespace."""
    if text is None:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove numbers
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def common_mistake(unit):
    if unit in allowed_units:
        return unit
    if unit.replace('ter', 'tre') in allowed_units:
        return unit.replace('ter', 'tre')
    if unit.replace('feet', 'foot') in allowed_units:
        return unit.replace('feet', 'foot')
    return unit

def parse_string(s):
    s_stripped = "" if s is None or str(s) == 'nan' else s.strip()
    if s_stripped == "":
        return None, None
    pattern = re.compile(r'^-?\d+(\.\d+)?\s+[a-zA-Z\s]+$')
    if not pattern.match(s_stripped):
        raise ValueError("Invalid format in {}".format(s))
    parts = s_stripped.split(maxsplit=1)
    number = float(parts[0])
    unit = common_mistake(parts[1])
    if unit not in allowed_units:
        raise ValueError("Invalid unit [{}] found in {}. Allowed units: {}".format(
            unit, s, allowed_units))
    return number, unit

def create_placeholder_image(image_save_path):
    try:
        placeholder_image = Image.new('RGB', (100, 100), color='black')
        placeholder_image.save(image_save_path)
    except Exception as e:
        print(f"Error creating placeholder image: {e}")

def download_image(image_link, save_folder, retries=3, delay=3):
    if not isinstance(image_link, str):
        return

    filename = Path(image_link).name
    image_save_path = os.path.join(save_folder, filename)

    if os.path.exists(image_save_path):
        return

    for _ in range(retries):
        try:
            response = requests.get(image_link)
            if response.status_code == 200:
                with open(image_save_path, 'wb') as f:
                    f.write(response.content)
                return
            else:
                print(f"Failed to download {image_link}: Status code {response.status_code}")
        except Exception as e:
            print(f"Error downloading {image_link}: {e}")
            time.sleep(delay)
    
    create_placeholder_image(image_save_path)  # Create a black placeholder image for invalid links/images

def download_image_wrapper(image_link, download_folder):
    """Wrapper function to download images."""
    download_image(image_link, download_folder)

def download_images(train_csv, test_csv, download_folder, allow_multiprocessing=True):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # Load image links from CSV files
    train_data = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)
    
    # Concatenate image links from train and test data
    image_links = train_data['image_link'].tolist() + test_data['image_link'].tolist()
    
    if allow_multiprocessing:
        with multiprocessing.Pool(64) as pool:
            list(tqdm(pool.imap(partial(download_image_wrapper, download_folder=download_folder), image_links), total=len(image_links)))
    else:
        for image_link in tqdm(image_links, total=len(image_links)):
            download_image(image_link, download_folder)

def extract_image_features(image_path):
    """
    Extract features from the image.
    You can use pre-trained models or custom feature extraction techniques here.
    """
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return np.zeros((224 * 224 * 3,))  # Return a zero array if the image is missing

    try:
        image = Image.open(image_path)
        # Resize the image to a fixed size
        image = image.resize((224, 224))
        
        # Convert the image to a numpy array
        image_data = np.array(image)
        
        # Flatten the image data
        image_features = image_data.flatten()
        
        return image_features
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return np.zeros((224 * 224 * 3,))  # Return a zero array if there's an error