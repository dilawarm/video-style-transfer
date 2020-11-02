import time

import IPython.display as display
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf

from model import StyleContentModel
from preprocessing import load_img, imshow, clip_0_1, tensor_to_image
from utils import total_variation_loss

mpl.rcParams["figure.figsize"] = (12, 12)
mpl.rcParams["axes.grid"] = False


def style_content_loss(outputs):
    """
    :param outputs: Generated image
    :return: The sum of the style and content loss
    """
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


@tf.function()
def train_step(image):
    """
    :param image: Input image
    """
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        loss += total_variation_weight * tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


if __name__ == "__main__":
    content_path = "images/content.jpg"
    style_path = "images/style.jpg"

    content_image = load_img(content_path)
    style_image = load_img(style_path)

    plt.subplot(1, 2, 1)
    imshow(content_image, "Content Image")

    plt.subplot(1, 2, 2)
    imshow(style_image, "Style Image")

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

    image = tf.Variable(content_image)

    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    style_weight = 1e-2
    content_weight = 1e4

    total_variation_weight = 30

    image = tf.Variable(content_image)

    start = time.time()

    epochs = 10
    steps_per_epoch = 100

    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(image)
            print(".", end="")
        display.clear_output(wait=True)
        display.display(tensor_to_image(image))
        print("Train step: {}".format(step))

    end = time.time()
    print("Total time: {:.1f}".format(end - start))
