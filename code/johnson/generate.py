import tensorflow as tf
import utils
import time
from moviepy.video.io.VideoFileClip import VideoFileClip
from subprocess import check_output
import transform
import numpy as np
import cv2

STYLE_MODEL = "checkpoints/la_muse.ckpt"
CONTENT = "../videos/shrek.mp4"
OUTPUT = "../stylized_videos/shrek.avi"
MAX_SIZE = 1024
BATCH_SIZE = 4


def convert_to_video(images):
    frame_array = []
    path_out = OUTPUT
    size = 0

    for i in images:
        file_name = "temp.jpg"
        utils.save_image(i, file_name)
        img = cv2.imread(file_name)
        check_output(f"rm -rf {file_name}".split())
        height, width, layers = img.shape
        size = (width, height)

        frame_array.append(img)
    out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*"DIVX"), 30, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


if __name__ == "__main__":
    video = VideoFileClip(CONTENT, audio=False)
    styled = []

    transformer = transform.Transform()
    start_time = time.time()

    soft_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=soft_config)

    with tf.compat.v1.Session(config=soft_config) as sess:
        shape = (BATCH_SIZE, video.size[1], video.size[0], 3)
        image = tf.compat.v1.placeholder(tf.float32, shape=shape, name="image")
        pred = transformer.net(image)
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, STYLE_MODEL)
        images = np.zeros(shape, dtype=np.float32)

        def write(tot):
            for i in range(tot, BATCH_SIZE):
                images[i] = images[i - 1]
            pred_n = sess.run(pred, feed_dict={image: images})
            for i in range(tot):
                styled.append(np.clip(pred_n[i], 0, 255).astype(np.uint8))

        tot = 0
        for frame in video.iter_frames():
            images[tot] = frame
            tot += 1
            if tot == BATCH_SIZE:
                write(tot)
                tot = 0

        if tot != 0:
            write(tot)

    end_time = time.time()

    convert_to_video(styled)

    print(f"Execution time {end_time - start_time}")
    print("The video has been styled!")