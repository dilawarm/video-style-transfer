import tensorflow as tf
import numpy as np
import collections
import transform
import utils


class StyleTransferTrainer:
    def __init__(
        self,
        content_layer_ids,
        style_layer_ids,
        content_images,
        style_image,
        session,
        net,
        num_epochs,
        batch_size,
        content_weight,
        style_weight,
        tv_weight,
        learn_rate,
        save_path,
        check_period,
    ):

        self.net = net
        self.sess = session

        self.CONTENT_LAYERS = collections.OrderedDict(sorted(content_layer_ids.items()))
        self.STYLE_LAYERS = collections.OrderedDict(sorted(style_layer_ids.items()))

        self.x_list = content_images
        mod = len(content_images) % batch_size
        self.x_list = self.x_list[:-mod]
        self.y_s0 = style_image
        self.content_size = len(self.x_list)

        self.num_epochs = num_epochs
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.check_period = check_period

        self.save_path = save_path

        self.transform = transform.Transform()

        self.build_graph()

    def build_graph(self):
        self.batch_shape = (self.batch_size, 256, 256, 3)

        self.y_c = tf.compat.v1.placeholder(
            tf.float32, shape=self.batch_shape, name="content"
        )
        self.y_s = tf.compat.v1.placeholder(
            tf.float32, shape=self.y_s0.shape, name="style"
        )

        self.y_c_pre = self.net.preprocess(self.y_c)
        self.y_s_pre = self.net.preprocess(self.y_s)

        content_layers = self.net.feed_forward(self.y_c_pre, scope="content")
        self.Ps = {}
        for id in self.CONTENT_LAYERS:
            self.Ps[id] = content_layers[id]

        style_layers = self.net.feed_forward(self.y_s_pre, scope="style")
        self.As = {}
        for id in self.STYLE_LAYERS:
            self.As[id] = self.gram_matrix(style_layers[id])

        self.x = self.y_c / 255.0
        self.y_hat = self.transform.net(self.x)

        self.y_hat_pre = self.net.preprocess(self.y_hat)
        self.Fs = self.net.feed_forward(self.y_hat_pre, scope="mixed")

        L_content = 0
        L_style = 0
        for id in self.Fs:
            if id in self.CONTENT_LAYERS:

                F = self.Fs[id]
                P = self.Ps[id]

                (
                    b,
                    h,
                    w,
                    d,
                ) = F.get_shape()
                b = b
                N = h * w
                M = d

                w = self.CONTENT_LAYERS[id]

                L_content += w * 2 * tf.nn.l2_loss(F - P) / (b * N * M)

            elif id in self.STYLE_LAYERS:
                F = self.Fs[id]

                (
                    b,
                    h,
                    w,
                    d,
                ) = F.get_shape()
                b = b
                N = h * w
                M = d

                w = self.STYLE_LAYERS[id]

                G = self.gram_matrix(F, (b, N, M))
                A = self.As[id]

                L_style += w * 2 * tf.nn.l2_loss(G - A) / (b * (M ** 2))

        L_tv = self.total_variation_loss(self.y_hat)

        alpha = self.content_weight
        beta = self.style_weight
        gamma = self.tv_weight

        self.L_content = alpha * L_content
        self.L_style = beta * L_style
        self.L_tv = gamma * L_tv
        self.L_total = self.L_content + self.L_style + self.L_tv

        tf.summary.scalar("L_content", self.L_content)
        tf.summary.scalar("L_style", self.L_style)
        tf.summary.scalar("L_tv", self.L_tv)
        tf.summary.scalar("L_total", self.L_total)

    def total_variation_loss(self, img):
        b, h, w, d = img.get_shape()
        b = b
        h = h
        w = w
        d = d
        tv_y_size = (h - 1) * w * d
        tv_x_size = h * (w - 1) * d
        y_tv = tf.nn.l2_loss(img[:, 1:, :, :] - img[:, : self.batch_shape[1] - 1, :, :])
        x_tv = tf.nn.l2_loss(img[:, :, 1:, :] - img[:, :, : self.batch_shape[2] - 1, :])
        loss = 2.0 * (x_tv / tv_x_size + y_tv / tv_y_size) / b

        loss = tf.cast(loss, tf.float32)
        return loss

    def train(self):
        global_step = tf.compat.v1.train.get_or_create_global_step()

        trainable_variables = tf.compat.v1.trainable_variables()
        grads = tf.gradients(self.L_total, trainable_variables)

        optimizer = tf.compat.v1.train.AdamOptimizer(self.learn_rate)
        train_op = optimizer.apply_gradients(
            zip(grads, trainable_variables), global_step=global_step, name="train_step"
        )

        # merged_summary_op = tf.compat.v1.summary.merge_all()

        # summary_writer = tf.compat.v1.summary.FileWriter(
        # self.save_path, graph=tf.compat.v1.get_default_graph()
        # )

        self.sess.run(tf.compat.v1.global_variables_initializer())

        saver = tf.compat.v1.train.Saver()

        checkpoint_exists = True
        try:
            ckpt_state = tf.compat.v1.train.get_checkpoint_state(self.save_path)
        except tf.errors.OutOfRangeError as e:
            print(f"Cannot restore checkpoint: {e}")
            checkpoint_exists = False
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            print(f"No model to restore at {self.save_path}")
            checkpoint_exists = False

        if checkpoint_exists:
            tf.logging.info(f"Loading checkpoint {ckpt_state.model_checkpoint_path}")
            saver.restore(self.sess, ckpt_state.model_checkpoint_path)

        num_examples = len(self.x_list)

        if checkpoint_exists:
            iterations = self.sess.run(global_step)
            epoch = (iterations * self.batch_size) // num_examples
            iterations = iterations - epoch * (num_examples // self.batch_size)
        else:
            epoch = 0
            iterations = 0

        while epoch < self.num_epochs:
            while iterations * self.batch_size < num_examples:

                curr = iterations * self.batch_size
                step = curr + self.batch_size
                x_batch = np.zeros(self.batch_shape, dtype=np.float32)
                for j, img_p in enumerate(self.x_list[curr:step]):
                    x_batch[j] = utils.get_img(img_p, (256, 256, 3)).astype(np.float32)

                iterations += 1

                assert x_batch.shape[0] == self.batch_size

                _, L_total, L_content, L_style, L_tv, step = self.sess.run(
                    [
                        train_op,
                        self.L_total,
                        self.L_content,
                        self.L_style,
                        self.L_tv,
                        global_step,
                    ],
                    feed_dict={self.y_c: x_batch, self.y_s: self.y_s0},
                )

                print(f"epoch : {epoch}, iter : {step}, ")
                print(
                    f"L_total : {L_total}, L_content : {L_content}, L_style : {L_style}, L_tv : {L_tv}"
                )
                # break

                # summary_writer.add_summary(summary, iterations)

                if step % self.check_period == 0:
                    res = saver.save(self.sess, self.save_path + "/final.ckpt", step)

            epoch += 1
            iterations = 0
        print("Saving final model...")
        res = saver.save(self.sess, self.save_path + "/final.ckpt")

    def gram_matrix(self, tensor, shape=None):

        if shape is not None:
            B = shape[0]
            HW = shape[1]
            C = shape[2]
            CHW = C * HW

        else:
            B, H, W, C = map(lambda i: i, tensor.get_shape())
            HW = H * W
            CHW = W * H * C

        feats = tf.reshape(tensor, (B, HW, C))

        feats_T = tf.transpose(feats, perm=[0, 2, 1])

        gram = tf.matmul(feats_T, feats) / CHW

        return gram