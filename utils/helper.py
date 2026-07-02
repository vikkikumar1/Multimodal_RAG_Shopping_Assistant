import json
import pandas as pd
import numpy as np
from PIL import Image
import io
import base64

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def clean_text(text):
    if not isinstance(text, str):
        return ""
    return text.strip().replace('\n', ' ').replace('\r', '')