from PIL import Image
import os

def resize_image(image_path, max_size=(512,512)):
    img = Image.open(image_path)
    img.thumbnail(max_size)
    return img

def save_uploaded_image(uploaded_file, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path