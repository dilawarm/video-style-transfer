import cv2
import tensorflow as tf
import sys
import numpy as np
from subprocess import check_output

sys.path.append("..")
from utils.utils import tensor_to_image


def preserve_colors(stylized_images, yuvs):
    """
    Method for preserving color of a video using YUV
    """
    sty_images = []

    for idx in range(len(yuvs)):
        sty = stylized_images[idx]
        content_yuv = yuvs[idx]

        sty_yuv = cv2.cvtColor(np.float32(sty[0].numpy()), cv2.COLOR_RGB2YUV)

        sty_yuv[:, :, 1:3] = content_yuv[:, :, 1:3]

        img = cv2.cvtColor(sty_yuv, cv2.COLOR_YUV2RGB)
        img = tf.convert_to_tensor([img], dtype=tf.float32)

        sty_images.append(img)
    return sty_images


def convert_to_video(images, frame_interval, name="video"):
    """
    Converts array of images to a video
    """
    frame_array = []
    path_out = "../stylized_videos/" + name + ".mp4"
    size = 0

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
