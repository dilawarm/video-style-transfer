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
import tensorflow_addons as tfa
from tqdm import tqdm
from subprocess import check_output

mpl.rcParams["figure.figsize"] = (12, 12)
mpl.rcParams["axes.grid"] = False
os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "COMPRESSED"

frame_interval = 0.1
loss_tolerance = 1000

TOL = 0.003

keep_colors = False


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
        return PIL.Image.fromarray(tensor)


def load_img(path_to_img):
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

    yuv = cv2.cvtColor(np.float32(img[0].numpy()), cv2.COLOR_RGB2YUV)

    return img, yuv


def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


def gram_matrix(input_tensor):
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (
            outputs[: self.num_style_layers],
            outputs[self.num_style_layers :],
        )

        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {
            content_name: value
            for content_name, value in zip(self.content_layers, content_outputs)
        }

        style_dict = {
            style_name: value
            for style_name, value in zip(self.style_layers, style_outputs)
        }

        return {"content": content_dict, "style": style_dict}


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def high_pass_x_y(image):
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

    return x_var, y_var


def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))


def get_flow(image1, image2):
    flow = False
    print(image1.shape)
    return cv2.optflow.createOptFlow_DeepFlow().calc(
        cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY),
        flow,
    )
    """
    return cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY),
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
        )
  """


def load_video(path_to_video):
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
            check_output(f"rm -rf {image_path}".split())

    return images, yuvs


def convert_to_video(images):
    frame_array = []
    path_out = "./stylized_videos/video.avi"
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


def getFrame(sec, vidcap):
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames, image = vidcap.read()

    return hasFrames, image


def check_disocclusion(avg_flow, back_flow):
    sum = avg_flow + back_flow
    return (
        np.sqrt(np.linalg.norm(sum)) ** 2
        > 0.01 * (np.linalg.norm(avg_flow) ** 2 + np.linalg.norm(back_flow) ** 2) + 0.5
    )


def detect_motion_boundary(u, v):
    return (u ** 2 + v ** 2) > (0.01 * np.sqrt(u ** 2 + v ** 2) ** 2) + 0.002


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
        temporal_loss = tf.add_n([tf.reduce_mean(((image - omega) * c) ** 2)])
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


def preserve_colors(stylized_images, yuvs):
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


if __name__ == "__main__":
    style_path = "images/style.jpg"
    style_image, _ = load_img(style_path)

    images, yuvs = load_video("videos/cat3.mp4")

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
        temporal_weight = 5e4  # random veri bare, vi må teste ut ulike verdier for den

        total_variation_weight = 30

        """
        if len(styled_images) > 0:
            content_image = tf.where(
                content_image - images[idx - 1] < TOL, styled_images[-1], content_image
            )
        """

        c = np.ones(images[idx].shape)
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

            for x in range(len(average_flow[0])):
                for y in range(len(average_flow[0][x])):
                    if check_disocclusion(
                        average_flow[0][x][y], backward_flow[0][x][y]
                    ):  # or detect_motion_boundary(backward_flow[0][x][y][0], backward_flow[0][x][y][1]):
                        c[0][x][y] = np.zeros(3)
            content_image = tfa.image.dense_image_warp(styled_images[-1], flow)
            omega = tfa.image.dense_image_warp(styled_images[-1], flow)

        print("content to warped")

        display.display(tensor_to_image(images[idx]))
        display.display(tensor_to_image(omega))
        display.display(tensor_to_image(c))

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

                tempimg, done = train_step(image, omega, c, idx)
                if done and step > 100:
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
    if keep_colors:
        styled_images = preserve_colors(styled_images, yuvs)
    convert_to_video(styled_images)