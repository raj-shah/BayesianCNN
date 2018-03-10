import logging
import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers
from get_data import get_data_set
from Model import Model

logging.basicConfig(filename='log.out', level=logging.INFO)
logger = logging.getLogger('train')

if __name__ == '__main__':
    
    batch_size = 128
    num_iter = 10 ** 5
    lr = 0.01
    decay_steps = 1
    decay_rate = 0.0001
    power = 0.75
    checkpoint_freq = 1000

    train_x, train_y, train_l = get_data_set("train")
    test_x, test_y, test_l = get_data_set("test")

    # hard coding for now, replace these
    checkpoint_path = "./checkpoints/model"
    summary_dir = "./summaries/"

    with tf.Session() as sess:
        model = Model()

        global_step = 0
        saver = tf.train.Saver(max_to_keep = None)
        summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

        sess.run(tf.global_variables_initializer())
        saver.save(sess, checkpoint_path, global_step=global_step)

        while global_step < num_iter:
            randidx = np.random.randint(len(train_x), size=batch_size)
            batch_xs = train_x[randidx]
            batch_ys = train_y[randidx]

            summary, loss, acc, _ = sess.run([model.merged, model.loss, model.acc, model.train_op], feed_dict={
                model.x_: batch_xs,
                model.y: batch_ys,
                model.lr: lr
            })

            # decayed_learning_rate = learning_rate / (1 + decay_rate * global_step / decay_step)

            global_step += 1
            lr = lr / ((1 + decay_rate * (global_step / decay_steps)) ** power)
            summary_writer.add_summary(summary, global_step)
            summary_writer.flush()

            logger.info("Global step: {}    Loss: {}    Accuracy: {}".format(global_step, loss, acc))

            if global_step % checkpoint_freq == 0:
                val_acc = sess.run([model.val_acc], feed_dict={
                    model.x_: test_x,
                    model.y: test_y
                })
                logger.info("EVALUATION STEP:\tTest Accuracy: {}".format(val_acc))
                saver.save(sess, checkpoint_path, global_step=global_step)




