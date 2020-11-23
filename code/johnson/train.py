import tensorflow as tf
import numpy as np
import utils
import vgg19
import os
import style_transfer

VGG_MODEL = "pre_trained_model/imagenet-vgg-verydeep-19.mat"
TRAINDB_PATH = "../../../media/datasets/train2014"
STYLE = "style/vangogh.jpg"
OUTPUT = "models"

CONTENT_WEIGHT = 7.5e0
STYLE_WEIGHT = 5e2
TV_WEIGHT = 2e2

CONTENT_LAYERS = ["relu4_2"]
STYLE_LAYERS = ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]
CONTENT_LAYER_WEIGHTS = [1.0]
STYLE_LAYER_WEIGHTS = [0.2, 0.2, 0.2, 0.2, 0.2]

LEARN_RATE = 1e-3
NUM_EPOCHS = 2
BATCH_SIZE = 4

tf.compat.v1.disable_eager_execution()

if __name__ == "__main__":
    model_file_path = VGG_MODEL
    vgg_net = vgg19.VGG19(model_file_path)

    content_images = utils.get_files(TRAINDB_PATH)

    style_image = utils.load_image(STYLE)

    CONTENT_LAYERS_DICT = {}
    for layer, weight in zip(CONTENT_LAYERS, CONTENT_LAYER_WEIGHTS):
        CONTENT_LAYERS_DICT[layer] = weight

    STYLE_LAYERS_DICT = {}
    for layer, weight in zip(STYLE_LAYERS, STYLE_LAYER_WEIGHTS):
        STYLE_LAYERS_DICT[layer] = weight

    sess = tf.compat.v1.Session(
        config=tf.compat.v1.ConfigProto(allow_soft_placement=True)
    )

    trainer = style_transfer.Fit(
        session=sess,
        content_layer_ids=CONTENT_LAYERS_DICT,
        style_layer_ids=STYLE_LAYERS_DICT,
        content_images=content_images,
        style_image=np.reshape(style_image, (1,) + style_image.shape),
        net=vgg_net,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        content_weight=CONTENT_WEIGHT,
        style_weight=STYLE_WEIGHT,
        tv_weight=TV_WEIGHT,
        learn_rate=LEARN_RATE,
        save_path=OUTPUT,
    )

    trainer.train()

    sess.close()
    print("Training finished!")
