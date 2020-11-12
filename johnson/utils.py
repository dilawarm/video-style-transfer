import numpy as np
import PIL.Image
import os
import imageio
from skimage.transform import resize


def get_img(src, img_size=False):
    img = imageio.imread(src, pilmode="RGB")
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img, img, img))
    if img_size != False:
        img = resize(img, img_size)
    return img


def get_files(img_dir):
    files = list_files(img_dir)
    return list(map(lambda x: os.path.join(img_dir, x), files))


def list_files(in_path):
    files = []
    for (_, _, filenames) in os.walk(in_path):
        files.extend(filenames)
        break
    return files


def load_image(filename, shape=None, max_size=None):
    image = PIL.Image.open(filename)

    if max_size is not None:
        factor = float(max_size) / np.max(image.size)

        size = np.array(image.size) * factor

        size = size.astype(int)

        image = image.resize(size, PIL.Image.LANCZOS)

    if shape is not None:
        image = image.resize(shape, PIL.Image.LANCZOS)

    return np.float32(image)


def save_image(image, filename):
    image = np.clip(image, 0.0, 255.0)

    image = image.astype(np.uint8)

    with open(filename, "wb") as file:
        PIL.Image.fromarray(image).save(file, "jpeg")