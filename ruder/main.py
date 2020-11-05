import time
import os

import IPython.display as display
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

from model import StyleContentModel
from preprocessing import (
    load_img,
    imshow,
    clip_0_1,
    tensor_to_image,
    load_video,
    convert_to_video,
)
from utils import total_variation_loss

TOL = 0.05
loss_tolerance = 10000

mpl.rcParams["figure.figsize"] = (12, 12)
mpl.rcParams["axes.grid"] = False


def loss_function(prev, image, idx, warped_image):
    """
    :param last_image: the previous generated frame
    :param outputs: Generated image
    :return: The sum of the style and content loss
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

    temporal_loss = 0
    if idx > 0:
        temporal_loss = tf.add_n([tf.reduce_mean((image - warped_image) ** 2)])
        temporal_loss *= temporal_weight

    loss = style_loss + content_loss + temporal_loss
    return loss


def train_step(prev, image, idx):
    """
    :param last_image: the previous frame generated
    :param image: Input image
    :return: the current image after the training step
    """
    done = False
    with tf.GradientTape() as tape:
        if idx > 0:
            warped_image = tf.tfa.image.dense_image_warp(image, flow[idx - 1])
        loss = loss_function(image, warped_image, idx)
        loss += total_variation_weight * tf.image.total_variation(image)
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))

    if len(losses) > 0 and abs(loss.numpy() - losses[-1].numpy()) < loss_tolerance:
        done = True

    losses.append(loss)

    return image, done


if __name__ == "__main__":
    style_path = "images/style.jpg"
    style_image = load_img(style_path)

    images, flow = load_video("videos/cat3.mp4")

    styled_images = []
    losses = []
    start = time.time()

    for idx, img in enumerate(images):

        content_image = img

        content_layers = ["block5_conv2"]

        style_layers = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]

        num_content_layers = len(content_layers)
        num_style_layers = len(style_layers)

        extractor = StyleContentModel(style_layers, content_layers)

        style_targets = extractor(style_image)["style"]
        content_targets = extractor(content_image)["content"]

        opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

        style_weight = 1e-2
        content_weight = 1e4
        temporal_weight = 1e5  # random veri bare, vi må teste ut ulike verdier for den

        total_variation_weight = 30

        if len(styled_images) > 0:
            content_image = tf.where(
                content_image - images[idx - 1] < TOL, styled_images[-1], content_image
            )

        image = tf.Variable(content_image)

        start = time.time()

        epochs = 10

        steps_per_epoch = 100

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

                tempimg, done = train_step(prev, image, idx)
                if done:
                    optim_done = True
                    break
                print(".", end="")
            display.clear_output(wait=True)
            display.display(tensor_to_image(image))
            print("Train step: {}".format(step))

        end = time.time()
        print("Total time: {:.1f}".format(end - start))
        print(f"{idx+1}/{len(images)} frames processed")

        styled_images.append(image)

    print("Total time: {:.1f}".format(end - start))
    convert_to_video(styled_images)