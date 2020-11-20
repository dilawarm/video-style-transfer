import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import tensorflow as tf
from subprocess import check_output

def getFrame(sec, vidcap):
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames, image = vidcap.read()

    return hasFrames, image

def load_img(path_to_img):
    max_dim = 1024
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]

    yuv = cv2.cvtColor(np.float32(img[0].numpy()), cv2.COLOR_RGB2YUV)

    return img, yuv

def load_video(path_to_video, frame_interval):
    vidcap = cv2.VideoCapture(path_to_video)
    sec = 0
    count = 0
    success = getFrame(sec, vidcap)
    images = []
    yuvs = []
    while success:
        count += 1
        sec += frame_interval
        sec = round(sec, 2)
        success, image = getFrame(sec, vidcap)
        if success:
            image_path = str(count) + ".jpg"
            cv2.imwrite(image_path, image)

            i, j = load_img(image_path)
            images.append(i)
            yuvs.append(j)
            #check_output(f"rm -rf {image_path}".split())

    return images, yuvs

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
