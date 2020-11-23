import os
import time
from subprocess import check_output

import cv2 as cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import tensorflow as tf

import sys

sys.path.append("..")

from utils.preprocessing import load_img, load_video, clip_0_1
from utils.postprocessing import convert_to_video, preserve_colors
from utils.model import StyleContentModel

mpl.rcParams["figure.figsize"] = (12, 12)
mpl.rcParams["axes.grid"] = False


fps = 10.0
frame_interval = 1 / fps
loss_tolerance = 0.002


def loss_function(image):
    """
    Method that calculates style loss and content loss.
    """
    outputs = extractor(image)
    style_outputs = outputs["style"]
    content_outputs = outputs["content"]

    style_loss = tf.add_n(
        [
            tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
            for name in style_outputs.keys()
        ]
    )
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n(
        [
            tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
            for name in content_outputs.keys()
        ]
    )
    content_loss *= content_weight / num_content_layers

    loss = style_loss + content_loss
    return loss


def train_step(image):
    """
    :param last_image: the previous frame generated
    :param image: Input image
    :return: the current image after the training step
    """
    done = False
    with tf.GradientTape() as tape:
        loss = loss_function(image)
        loss += total_variation_weight * tf.image.total_variation(image)
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))

    if len(losses) > 50:
        done = True
        for i in range(-51, -2):
            if (losses[i] - losses[i + 1]) > loss_tolerance * losses[i]:
                done = False

    losses.append(loss)

    return image, done


if __name__ == "__main__":
    style_path = "../images/style.jpg"
    style_image, _ = load_img(style_path)

    images, yuvs = load_video("../videos/cat2.mp4", frame_interval)

    styled_images = []
    losses = []
    start = time.time()

    content_layers = ["block5_conv2"]

    style_layers = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1",
    ]

    epochs = 10
    steps_per_epoch = 100

    style_weight = 1e-2
    content_weight = 1e4
    total_variation_weight = 30

    extractor = StyleContentModel(style_layers, content_layers)

    style_targets = extractor(style_image)["style"]

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    for idx, img in enumerate(images):
        losses = []

        content_image = img

        content_targets = extractor(content_image)["content"]

        image = tf.Variable(content_image)

        start = time.time()
        step = 0
        optim_done = False
        for n in range(epochs):
            if optim_done:
                break
            for m in range(steps_per_epoch):
                step += 1
                tempimg, done = train_step(image)
                if done and step > 500:
                    optim_done = True
                    break
                print(".", end="")
            print("Train step: {}".format(step))

        end = time.time()
        print("Total time: {:.1f}".format(end - start))
        print(f"{idx+1}/{len(images)} frames processed")

        styled_images.append(image)

    convert_to_video(styled_images, frame_interval, name="video")
    styled_images = preserve_colors(styled_images, yuvs)
    convert_to_video(styled_images, frame_interval, name="video_preserved_colors")