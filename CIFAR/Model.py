import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers

class Model(object):
    def __init__(self, batch_size=128, learning_rate=0.01, num_labels=10, keep_prob=0.5, scope="model"):
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._num_labels = num_labels
        self._scope = scope
        self._keep_prob = keep_prob
        self._conv_hidden_dims = [192, 192]
        with tf.variable_scope(self._scope):
            self._build_model()

    def _build_net(self, x, reuse=False, trainable=True, scope="inference_net"):
        with tf.variable_scope(scope, reuse=reuse):
            out = x
            for i in range(len(self._conv_hidden_dims)):
                out = layers.conv2d(out, num_outputs=self._conv_hidden_dims[i], kernel_size=(5, 5),
                                    activation_fn=tf.nn.relu, trainable=trainable)
                out = layers.dropout(out, keep_prob=self._keep_prob, is_training=trainable)
                out = layers.max_pool2d(out, kernel_size=(2, 2))

            out = layers.flatten(out)
            out = layers.fully_connected(out, num_outputs=1000, activation_fn=tf.nn.relu, trainable=trainable)
            out = layers.dropout(out, keep_prob=self._keep_prob, is_training=trainable)
            logits = layers.fully_connected(out, self._num_labels, trainable=trainable)

        return logits

    def _build_model(self):
        self.x_ = tf.placeholder(tf.float32, shape=[None, 3072], name='x_')  # data gets loaded as a 32x32 vector
        x = tf.reshape(self.x_, [-1, 32, 32, 3], name='x')  # CIFAR dataset is shape 32,32,3
        self.y = tf.placeholder(tf.float32, shape=[None, self._num_labels], name='y')  # 10 labels
        # self.keep_prob = tf.placeholder(tf.float32, name='dropout_prob')
        self.lr = tf.placeholder(tf.float32, shape=(), name='lr')

        self.logits = self._build_net(x)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y)
        self.loss = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.MomentumOptimizer(self.lr, momentum=0.9, use_nesterov=True)
        self.train_op = optimizer.minimize(loss=self.loss)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1)), dtype=tf.float32))

        # for eval steps
        self.val_logits = self._build_net(x, reuse=True, trainable=False)
        self.val_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.val_logits, 1), tf.argmax(self.y, 1)), dtype=tf.float32))

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.acc)
        self.merged = tf.summary.merge_all()
