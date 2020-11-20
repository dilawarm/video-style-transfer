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

import sys
sys.path.append("..")
from utils.postprocessing import preserve_colors, convert_to_video
from utils.preprocessing import getFrame, load_img, load_video
from utils.utils import tensor_to_image, imshow, total_variation_loss
from model import StyleContentModel

mpl.rcParams["figure.figsize"] = (12, 12)
mpl.rcParams["axes.grid"] = False
os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "COMPRESSED"

fps = 20.0
frame_interval = 1.0/fps
loss_tolerance = 0.003


J = [1]

TOL = 0.0025


def get_flow(image1, image2):
    flow = False
    return cv2.optflow.createOptFlow_DeepFlow().calc(
        cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY),
        flow,
    )

def check_disocclusion(avg_flow, back_flow):
    sum = avg_flow + back_flow
    return (
        np.sqrt(np.linalg.norm(sum)) ** 2
        > 0.01 * (np.linalg.norm(avg_flow) ** 2 + np.linalg.norm(back_flow) ** 2) + 0.5
    )


def detect_motion_boundary(u_1, u_2, v_1, v_2):
    return (u_2 - u_1) **2  + (v_2 - v_1) ** 2 > (0.01 * np.sqrt(u_1 ** 2 + v_1 ** 2) ** 2) + 0.002

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res



def create_average_flow(flow, backward_flow):
  average_flow = np.zeros(flow.shape)
  print(average_flow.shape)
  for x in range(len(flow[0])):
    for y in range(len(flow[0][x])):
      x_index = int(round(x + backward_flow[0][x][y][1]))
      y_index = int(round(y + backward_flow[0][x][y][0]))
      if x_index < len(flow[0]) and y_index < len(flow[0][x]):
        average_flow[0][x][y] = flow[0][
                                        int(round(x + backward_flow[0][x][y][1]))
                                        ][
                                          int(round(y + backward_flow[0][x][y][0]))
                                          ]
  return average_flow



def create_c_matrix(idx, j, average_flow, backward_flow):
  c = np.ones(images[idx - j].shape)
  for x in range(len(average_flow[0])):
    for y in range(len(average_flow[0][x])):
      if check_disocclusion(average_flow[0][x][y], backward_flow[0][x][y]) or (x < len(backward_flow[0]) - 1 and y < len(backward_flow[0][x]) -1 and detect_motion_boundary(
                                    backward_flow[0][x][y][1], 
                                    backward_flow[0][x + 1][y][1],  
                                    backward_flow[0][x][y][0], 
                                    backward_flow[0][x][y+1][0])):
        c[0][x][y] = np.zeros(3)
  return c
  


def create_flow_lists(idx):
  flow_list = {}
  backward_flow_list = {}
  average_flow_list = {}

  for j in J:
    if idx - j < 0:
      break
    flow_list[j] = tf.convert_to_tensor([get_flow(images[idx - j][0].numpy() * 255, images[idx][0].numpy() * 255)]).numpy()
    backward_flow_list[j] = tf.convert_to_tensor([get_flow(images[idx][0].numpy() * 255, images[idx - j][0].numpy() * 255)]).numpy()
    average_flow_list[j] = create_average_flow(flow_list[j], backward_flow_list[j])
  
  return flow_list, backward_flow_list, average_flow_list

def create_c_list(idx, average_flow_list, backward_flow_list):
  c_list = {}
  for j in J:
    if idx - j < 0:
      break
    c_list[j] = create_c_matrix(idx, j, average_flow_list[j], backward_flow_list[j])

  return c_list

def create_c_long(idx, j, c_list):
  sum = np.zeros(c_list[j].shape)
  for k in J:
    if k == j:
      break
    sum += c_list[k]
  return np.maximum(c_list[j] - sum, np.zeros(c_list[j].shape))


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
        for j in J:
          if idx - j < 0:
            break
          temporal_loss += tf.add_n([tf.reduce_mean(((image - omega[j]) * c[j]) ** 2)])
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


    if len(losses) > 50:
      done = True
      for i in range(-51, -2):
        if (losses[i] - losses[i+1]) > loss_tolerance * losses[i]:
          done = False

    losses.append(loss)

    return image, done


if __name__ == "__main__":
    style_path = "images/style.jpg"
    style_image, _ = load_img(style_path)

    images, yuvs = load_video("videos/cat3.mp4", frame_interval)

    styled_images = []
    losses = []
    start = time.time()

    for idx, img in enumerate(images):
        losses = []

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


        ## 5 stilvekt er bra for 1280 max dim, 2 for 512 max
        style_weight = 1e-2
        content_weight = 1e4
        temporal_weight = 1e8  # random veri bare, vi må teste ut ulike verdier for den

        total_variation_weight = 30

        """
        if len(styled_images) > 0:
            content_image = tf.where(
                content_image - images[idx - 1] < TOL, styled_images[-1], content_image
            )
        """

        c = np.ones(images[idx].shape)
        omega = 0
        omega_2 = 0
        if len(styled_images) > 0:
            flow_dict, backward_flow_dict, average_flow_dict = create_flow_lists(idx)

            c_list = create_c_list(idx, average_flow_dict, backward_flow_dict)
            
            omega = {}
            c = {}
            for j in J:
              if idx - j < 0:
                break
              omega[j] = tf.convert_to_tensor([warp_flow(styled_images[idx - j][0].numpy(), flow_dict[j][0])])
              c[j] = create_c_long(idx, j, c_list)
            content_image = tf.convert_to_tensor([warp_flow(styled_images[-1][0].numpy(), flow_dict[1][0])])
            


        display.display(tensor_to_image(images[idx-1]))
        display.display(tensor_to_image(images[idx]))
        display.display(tensor_to_image(content_image))


        for j in J:
          if idx - j < 0:
            break
          display.display(tensor_to_image(c[j]))
 

        image = tf.Variable(content_image)

        start = time.time()

        epochs = 15

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
                if done and step > 500:
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
    convert_to_video(styled_images, name = "video")
    styled_images = preserve_colors(styled_images, yuvs)
    convert_to_video(styled_images, name = "video_preserved_colors")

    # husk å sjekke om average_flow eller noen av flow-matrisene kan transponeres for å få riktig disoccluded områder
