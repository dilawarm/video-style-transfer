from subprocess import check_output

import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import tensorflow as tf


frame_interval = 0.5


def tensor_to_image(tensor):
    """
    :param tensor: The tensor we want to transform into an image
    :return: An image
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img(path_to_img):
    """
    :param path_to_img: Path to the image we want to load
    :return: An image
    """
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def get_flow(image1, image2):
    return cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY),
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )


def load_video(path_to_video):
    vidcap = cv2.VideoCapture(path_to_video)
    sec = 0
    count = 0
    success = getFrame(sec, vidcap)
    images = []
    flow = []
    while success:
        count += 1
        sec += frame_interval
        sec = round(sec, 2)
        success, image = getFrame(sec, vidcap)
        prevImage = None
        if success:
            if prevImage:
                flow.append(get_flow(prevImage, image))
            image_path = str(count) + ".jpg"
            cv2.imwrite(image_path, image)
            images.append(load_img(image_path))
            check_output(f"rm -rf {image_path}".split())
            prevImage = image

    return images, flow


def convert_to_video(images):
    frame_array = []
    path_out = "./stylized_videos/video.avi"

    for i in images:
        file_name = "temp.jpg"
        tensor_to_image(i).save(file_name)
        img = cv2.imread(file_name)
        check_output(f"rm -rf {file_name}".split())
        height, width, layers = img.shape
        size = (width, height)

        frame_array.append(img)
    out = cv2.VideoWriter(
        path_out, cv2.VideoWriter_fourcc(*"DIVX"), 1 / frame_interval, size
    )

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def getFrame(sec, vidcap):
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames, image = vidcap.read()

    return hasFrames, image


def imshow(image, title=None):
    """
    :param image: The image we want to show
    :param title: Title of the image
    """
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


def clip_0_1(image):
    """
    :param image: An image
    :return: The image with pixel values bewteen 0 and 1
    """
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def high_pass_x_y(image):
    """
    :param image: An image
    :return: High frequency components of the image
    """
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

    return x_var, y_var
