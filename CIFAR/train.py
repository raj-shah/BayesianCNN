import logging
import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers
from get_data import get_data_set
from Model import Model

logging.basicConfig(filename='log.out', level=logging.INFO)
logger = logging.getLogger('train')


def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]
    

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
                
                test_batch_acc_total = 0
                test_batch_count = 0
                
                for test_feature_batch, test_label_batch in batch_features_labels(test_x, test_y, batch_size=100):
                    test_batch_acc_total += sess.run([model.val_acc], feed_dict={
                    model.x_: test_feature_batch,
                    model.y: test_label_batch
                    keep_prob: 1.0
                    })
                    test_batch_count += 1

                val_acc = test_batch_acc_total/test_batch_count
                
                logger.info("EVALUATION STEP:\tTest Accuracy: {}".format(val_acc))
                saver.save(sess, checkpoint_path, global_step=global_step)




