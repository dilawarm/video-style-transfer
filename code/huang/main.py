import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import PIL
import time
import functools
import IPython.display as display
import cv2
from tqdm import tqdm
from subprocess import check_output

import sys

from utilities import calc_mean_covariance, calc_wasserstein_distance
from model import StyleContentModel

sys.path.append("..")
from utils.preprocessing import load_img, load_video, clip_0_1
from utils.postprocessing import convert_to_video, preserve_colors
from utils.utils import tensor_to_image, create_flow_lists, create_c_list, warp_flow


mpl.rcParams["figure.figsize"] = (12, 12)
mpl.rcParams["axes.grid"] = False

fps = 10.0
frame_interval = 1.0 / fps
loss_tolerance = 0.0025


J = [1]

TOL = 0.003


def loss_function(image, idx, c, omega):
    """
    Method that calculates style loss, content loss and temporal loss.
    """
    outputs = extractor(image)
    style_outputs = outputs["style"]
    content_outputs = outputs["content"]

    style_losses = []
    for name in style_outputs.keys():
        img_mean, img_cov = calc_mean_covariance(style_outputs[name])
        style_losses.append(
            calc_wasserstein_distance(style_targets[name], img_mean, img_cov)
        )

    style_loss = tf.add_n(style_losses)
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n(
        [
            tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
            for name in content_outputs.keys()
        ]
    )
    content_loss *= content_weight / num_content_layers

    temporal_loss = 0
    if idx > 0:
        for j in J:
            if idx - j < 0:
                break
            temporal_loss += tf.add_n(
                [tf.reduce_mean(((image - omega[j]) * c[j]) ** 2)]
            )
        temporal_loss *= temporal_weight

    loss = style_loss + content_loss + temporal_loss
    return loss


def train_step(image, omega, c, idx):
    """
    :param last_image: the previous frame generated
    :param image: Input image
    :return: the current image after the training step
    """
    done = False
    with tf.GradientTape() as tape:
        loss = loss_function(image, idx, c, omega)
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

    content_layers = ["block5_conv2"]
    style_layers = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1",
    ]
    extractor = StyleContentModel(style_layers, content_layers)

    style_targets = extractor(style_image, target=True)["style"]

    styled_images = []
    losses = []
    start = time.time()
    epochs = 10
    steps_per_epoch = 100

    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    style_weight = 4
    content_weight = 1e4
    temporal_weight = 3e7

    total_variation_weight = 30

    for idx, img in enumerate(images):
        losses = []

        content_image = img

        num_content_layers = len(content_layers)
        num_style_layers = len(style_layers)

        content_targets = extractor(content_image)["content"]

        c = np.ones(images[idx].shape)
        omega = 0
        if len(styled_images) > 0:
            flow_dict, backward_flow_dict, average_flow_dict = create_flow_lists(idx, J)

            c_list = create_c_list(idx, average_flow_dict, backward_flow_dict, J)

            omega = {}
            c = {}
            for j in J:
                if idx - j < 0:
                    break
                omega[j] = tf.convert_to_tensor(
                    [warp_flow(styled_images[idx - j][0].numpy(), flow_dict[j][0])]
                )
                c[j] = create_c_long(idx, j, c_list)
            content_image = tf.convert_to_tensor(
                [warp_flow(styled_images[-1][0].numpy(), flow_dict[1][0])]
            )

        image = tf.Variable(content_image)

        start = time.time()

        step = 0

        optim_done = False
        for n in range(epochs):
            if optim_done:
                break
            for m in range(steps_per_epoch):
                step += 1
                prev = 0
                if idx > 0:
                    prev = styled_images[-1]

                tempimg, done = train_step(image, omega, c, idx)
                if done and step > 500:
                    optim_done = True
                    break
                print(".", end="")
            print("Train step: {}".format(step))

        end = time.time()
        print("Total time: {:.1f}".format(end - start))
        print(f"{idx+1}/{len(images)} frames processed")

        styled_images.append(image)

    # print("Total time: {:.1f}".format(end - start))
    convert_to_video(styled_images, frame_interval, name="video")
    styled_images = preserve_colors(styled_images, yuvs)
    convert_to_video(styled_images, frame_interval, name="video_preserved_colors")

    # husk å sjekke om average_flow eller noen av flow-matrisene kan transponeres for å få riktig disoccluded områder
