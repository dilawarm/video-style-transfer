import tensorflow as tf

from preprocessing import high_pass_x_y


def vgg_layers(layer_names):
    """
    :param layer_name: Intermediate layers for style and content
    :return: A vgg model that returns a list of intermediate output values
    """
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


def gram_matrix(input_tensor):
    """
    :param input_tensor: The feature vector we calculate the Gram matrix for.
    :return: The Gram matrix
    """
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)


def total_variation_loss(image):
    """
    :param image: An image
    :return: Total variational loss
    """
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))
