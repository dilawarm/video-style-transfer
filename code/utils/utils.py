import tensorflow as tf
import numpy as np
import PIL


def vgg_layers(layer_names):
    """
    Extracts the layers from the VGG-19 CNN specified in layer_names and returns the model.
    """
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


def high_pass_x_y(image):
    """
    Extracts high frequency artifacts from image
    """
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

    return x_var, y_var


def gram_matrix(input_tensor):
    """
    calculates and returns the gram-matrix of a tf tensor
    """
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)


def total_variation_loss(image):
    """
    calculates total variation loss on an image
    """
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))


def tensor_to_image(tensor):
    """
    converts a tensor to an image using the PIL library
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
        return PIL.Image.fromarray(tensor)


def imshow(image, title=None):
    """
    prints out an image
    """
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


def warp_flow(img, flow):
    """
    Warps an image given its flow values
    """
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


def create_average_flow(flow, backward_flow):
    """
    Calculates the average flow function given flow and backward flow.
    """
    average_flow = np.zeros(flow.shape)
    print(average_flow.shape)
    for x in range(len(flow[0])):
        for y in range(len(flow[0][x])):
            x_index = int(round(x + backward_flow[0][x][y][1]))
            y_index = int(round(y + backward_flow[0][x][y][0]))
            if x_index < len(flow[0]) and y_index < len(flow[0][x]):
                average_flow[0][x][y] = flow[0][
                    int(round(x + backward_flow[0][x][y][1]))
                ][int(round(y + backward_flow[0][x][y][0]))]
    return average_flow


def create_c_matrix(idx, j, average_flow, backward_flow):
    """
    Calculates the c matrix between image idx and image idx - j, given the average flow and backward flow.
    """
    c = np.ones(images[idx - j].shape)
    for x in range(len(average_flow[0])):
        for y in range(len(average_flow[0][x])):
            if check_disocclusion(average_flow[0][x][y], backward_flow[0][x][y]) or (
                x < len(backward_flow[0]) - 1
                and y < len(backward_flow[0][x]) - 1
                and detect_motion_boundary(
                    backward_flow[0][x][y][1],
                    backward_flow[0][x + 1][y][1],
                    backward_flow[0][x][y][0],
                    backward_flow[0][x][y + 1][0],
                )
            ):
                c[0][x][y] = np.zeros(3)
    return c


def create_flow_lists(idx, J):
    """
    calculates and returns all the flow functions to use in c-matrix based on the elements in J
    """
    flow_list = {}
    backward_flow_list = {}
    average_flow_list = {}

    for j in J:
        if idx - j < 0:
            break
        flow_list[j] = tf.convert_to_tensor(
            [get_flow(images[idx - j][0].numpy() * 255, images[idx][0].numpy() * 255)]
        ).numpy()
        backward_flow_list[j] = tf.convert_to_tensor(
            [get_flow(images[idx][0].numpy() * 255, images[idx - j][0].numpy() * 255)]
        ).numpy()
        average_flow_list[j] = create_average_flow(flow_list[j], backward_flow_list[j])

    return flow_list, backward_flow_list, average_flow_list


def create_c_list(idx, average_flow_list, backward_flow_list, J):
    """
    Creates and returns all the c values for the given image idx, using the produced flow-functions and specified long term frames in J
    """
    c_list = {}
    for j in J:
        if idx - j < 0:
            break
        c_list[j] = create_c_matrix(idx, j, average_flow_list[j], backward_flow_list[j])

    return c_list


def create_c_long(idx, j, c_list):
    """
    creates and returns the value of c_long between frame idx and frame idx - j, using the list of c-matrixes
    """
    sum = np.zeros(c_list[j].shape)
    for k in J:
        if k == j:
            break
        sum += c_list[k]
    return np.maximum(c_list[j] - sum, np.zeros(c_list[j].shape))
