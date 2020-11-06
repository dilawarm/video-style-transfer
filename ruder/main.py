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
    get_flow,
)
from utils import total_variation_loss

frame_interval = 0.1
TOL = 0.05
loss_tolerance = 10000

mpl.rcParams["figure.figsize"] = (12, 12)
mpl.rcParams["axes.grid"] = False


def loss_function(image, idx, c, omega):
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
        temporal_loss = tf.add_n([tf.reduce_mean((image - omega) ** 2)])
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
        warped_image = 0
        loss = loss_function(image, idx, c, omega)
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

    images, warped = load_video("videos/cat3.mp4")

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
        temporal_weight = 4e4  # random veri bare, vi må teste ut ulike verdier for den

        total_variation_weight = 30

        """
        if len(styled_images) > 0:
            content_image = tf.where(
                content_image - images[idx - 1] < TOL, styled_images[-1], content_image
            )
        """

        c = tf.zeros(image.shape)
        omega = 0
        if len(styled_images) > 0:
            flow = tf.convert_to_tensor(
                [
                    get_flow(
                        images[idx - 1][0].numpy() * 255, images[idx][0].numpy() * 255
                    )
                ]
            )
            backward_flow = tf.convert_to_tensor(
                [
                    get_flow(
                        images[idx][0].numpy() * 255, images[idx - 1][0].numpy() * 255
                    )
                ]
            )

            average_flow = np.zeros(flow.shape)
            flow = flow.numpy()
            backward_flow = backward_flow.numpy()

            for x in range(len(flow[0])):
                for y in range(len(flow[0][x])):
                    x_index = int(round(x + backward_flow[0][x][y][0]))
                    y_index = int(round(y + backward_flow[0][x][y][1]))
                    if x_index < len(flow[0]) and y_index < len(flow[0][x]):
                        average_flow[0][x][y] = flow[0][
                            int(round(x + backward_flow[0][x][y][0]))
                        ][int(round(y + backward_flow[0][x][y][1]))]

            c = tf.where(
                (
                    abs(average_flow + backward_flow) ** 2
                    > 0.01 * (abs(average_flow) ** 2 + abs(backward_flow) ** 2) + 0.5
                ),
                tf.zeros(average_flow.shape),
                tf.ones(average_flow.shape),
            )

            omega = tfa.image.dense_image_warp(styled_images[-1], average_flow)

        print("content to warped")
        display.display(tensor_to_image(images[idx - 1]))
        display.display(tensor_to_image(images[idx]))
        display.display(tensor_to_image(omega))

        image = tf.Variable(content_image)

        start = time.time()

        epochs = 4

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

                tempimg, done = train_step(image, omega, c, idx)
                if done:
                    optim_done = True
                    break
                print(".", end="")
            print("Train step: {}".format(step))

        end = time.time()
        print("Total time: {:.1f}".format(end - start))
        display.display(tensor_to_image(image))
        print(f"{idx+1}/{len(images)} frames processed")

        styled_images.append(image)

    # print("Total time: {:.1f}".format(end - start))
    convert_to_video(styled_images)