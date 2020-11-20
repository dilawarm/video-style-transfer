import tensorflow as tf
import sys
from utilities import calc_2_moments

sys.path.append("..")
from utils.utils import vgg_layers


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs, target=False):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (
            outputs[: self.num_style_layers],
            outputs[self.num_style_layers :],
        )

        styles_output_layers = []
        if target:
            for style_output in style_outputs:
                mean, cov = calc_2_moments(style_output)

                eigvals, eigvects = tf.linalg.eigh(cov)
                eigroot_mat = tf.linalg.diag(tf.sqrt(tf.maximum(eigvals, 0.0)))
                root_cov = tf.matmul(
                    tf.matmul(eigvects, eigroot_mat), eigvects, transpose_b=True
                )

                tr_cov = tf.reduce_sum(tf.maximum(eigvals, 0))

                styles_output_layers.append((mean, tr_cov, root_cov))
            style_outputs = styles_output_layers

        content_dict = {
            content_name: value
            for content_name, value in zip(self.content_layers, content_outputs)
        }

        style_dict = {
            style_name: value
            for style_name, value in zip(self.style_layers, style_outputs)
        }

        return {"content": content_dict, "style": style_dict}