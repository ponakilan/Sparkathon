import numpy as np
from PIL import Image

def forecast() -> np.ndarray:
    return np.random.rand((15))

def explain():
    imarray = np.random.rand(224, 512, 3) * 255
    image = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
    return image, image, "Hello!!"
