# Koden er basert på https://github.com/lengstrom/fast-style-transfer/blob/master/src/transform.py

import tensorflow as tf


class ImageTransformationNetwork:
    def __init__(self):
        self.reuse = None

    def net(self, image):
        image_p = self.reflection_padding(image)
        conv1 = self.conv_layer(image_p, 32, 9, 1, name="conv1")
        conv2 = self.conv_layer(conv1, 64, 3, 2, name="conv2")
        conv3 = self.conv_layer(conv2, 128, 3, 2, name="conv3")
        resid1 = self.residual_block(conv3, 3, name="resid1")
        resid2 = self.residual_block(resid1, 3, name="resid2")
        resid3 = self.residual_block(resid2, 3, name="resid3")
        resid4 = self.residual_block(resid3, 3, name="resid4")
        resid5 = self.residual_block(resid4, 3, name="resid5")
        conv_t1 = self.conv_transpose_layer(resid5, 64, 3, 2, name="convt1")
        conv_t2 = self.conv_transpose_layer(conv_t1, 32, 3, 2, name="convt2")
        conv_t3 = self.conv_layer(conv_t2, 3, 9, 1, relu=False, name="convt3")
        preds = (tf.nn.tanh(conv_t3) + 1) * (255.0 / 2)
        return preds

    def reflection_padding(self, net):
        return tf.pad(net, [[0, 0], [40, 40], [40, 40], [0, 0]], "REFLECT")

    def conv_layer(
        self,
        net,
        num_filters,
        filter_size,
        strides,
        padding="SAME",
        relu=True,
        name=None,
    ):
        weights_init = self.initialize_weights(net, num_filters, filter_size, name=name)
        strides_shape = [1, strides, strides, 1]
        net = tf.nn.conv2d(net, weights_init, strides_shape, padding=padding)
        net = self.instance_normalization(net, name=name)
        if relu:
            net = tf.nn.relu(net)

        return net

    def conv_transpose_layer(self, net, num_filters, filter_size, strides, name=None):
        weights_init = self.initialize_weights(
            net, num_filters, filter_size, transpose=True, name=name
        )

        batch_size, rows, cols, _ = [i for i in net.get_shape()]
        new_rows, new_cols = int(rows * strides), int(cols * strides)

        new_shape = [batch_size, new_rows, new_cols, num_filters]
        tf_shape = tf.stack(new_shape)
        strides_shape = [1, strides, strides, 1]

        net = tf.nn.conv2d_transpose(
            net, weights_init, tf_shape, strides_shape, padding="SAME"
        )
        net = self.instance_normalization(net, name=name)
        return tf.nn.relu(net)

    def residual_block(self, net, filter_size=3, name=None):
        batch, rows, cols, channels = [i for i in net.get_shape()]
        tmp = self.conv_layer(
            net, 128, filter_size, 1, padding="VALID", relu=True, name=name + "_1"
        )
        return self.conv_layer(
            tmp, 128, filter_size, 1, padding="VALID", relu=False, name=name + "_2"
        ) + tf.slice(net, [0, 2, 2, 0], [batch, rows - 4, cols - 4, channels])

    def instance_normalization(self, net, name=None):
        _, _, _, channels = [i for i in net.get_shape()]
        var_shape = [channels]
        mu, sigma_sq = tf.compat.v1.nn.moments(net, [1, 2], keep_dims=True)
        with tf.compat.v1.variable_scope(name, reuse=self.reuse):
            shift = tf.compat.v1.get_variable(
                "shift", initializer=tf.zeros(var_shape), dtype=tf.float32
            )
            scale = tf.compat.v1.get_variable(
                "scale", initializer=tf.ones(var_shape), dtype=tf.float32
            )
        epsilon = 1e-3
        normalized = (net - mu) / (sigma_sq + epsilon) ** (0.5)
        return scale * normalized + shift

    def initialize_weights(
        self, net, out_channels, filter_size, transpose=False, name=None
    ):
        _, _, _, in_channels = [i for i in net.get_shape()]
        if not transpose:
            weights_shape = [filter_size, filter_size, in_channels, out_channels]
        else:
            weights_shape = [filter_size, filter_size, out_channels, in_channels]
        with tf.compat.v1.variable_scope(name, reuse=self.reuse):
            weights_init = tf.compat.v1.get_variable(
                "weight",
                shape=weights_shape,
                initializer=tf.initializers.GlorotUniform(),
                dtype=tf.float32,
            )
        return weights_init