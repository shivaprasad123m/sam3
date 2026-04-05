import os
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import torch

sam3_root = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(sam3_root, 'assets', 'uncertainImages', 'truck.jpg')
image = Image.open(image_path)
print('image size', image.size)
