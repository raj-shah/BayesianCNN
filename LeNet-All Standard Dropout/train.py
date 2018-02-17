
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from Model import Model



FLAGS = tf.app.flags.FLAGS

def train():
    model = Model()

    with tf.Graph().as_default():
        # Load training data and use test as evaluation set in one hot format
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        # images = mnist.train.images # Returns np.array
        # labels = np.asarray(mnist.train.labels, dtype=np.int32)
        # val_images = mnist.test.images # Returns np.array
        # val_labels = np.asarray(mnist.test.labels, dtype=np.int32)

        x_ = tf.placeholder(tf.float32, shape=[None, 784]) #data gets loaded as a 28x8 vector
        x = tf.reshape(x_,[-1,28,28,1],name = 'x')   #mnist dataset is shape 28,28,1
        y = tf.placeholder(shape=[None, 10], dtype=tf.float32, name='y') #10 labels
        keep_prob = tf.placeholder(tf.float32, name='dropout_prob')
        global_step = tf.train.get_or_create_global_step()

        logits = model.inference(x, keep_prob=keep_prob)
        loss = model.loss(logits=logits, labels=y)

        accuracy = model.accuracy(logits, y)
        summary_op = tf.summary.merge_all()
        train_op = model.train(loss, global_step=global_step)

        init = tf.global_variables_initializer() 
        saver = tf.train.Saver()

        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
            sess.run(init)
            for i in range(FLAGS.num_iter):
                batch = mnist.train.next_batch(FLAGS.batch_size)

                _, cur_loss, summary = sess.run([train_op, loss, summary_op],
                                                feed_dict={x_: batch[0], y: batch[1], keep_prob: 0.5})
                writer.add_summary(summary, i)
                if i % 1000 == 0:
                    print('Iter {} Accuracy: {}'.format(i, cur_loss))
                    validation_accuracy = accuracy.eval(feed_dict={x_: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}) #evaluate test set
                    print('Test_Accuracy: {}'.format(validation_accuracy))

                if i == FLAGS.num_iter - 1:
                    saver.save(sess, FLAGS.checkpoint_file_path, global_step)

def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.flags.DEFINE_integer('batch_size', 64, 'size of training batches')
    tf.app.flags.DEFINE_integer('num_iter', 10000000, 'number of training iterations')
    tf.app.flags.DEFINE_string('checkpoint_file_path', 'checkpoints/model.ckpt-10000', 'path to checkpoint file')
    tf.app.flags.DEFINE_string('summary_dir', 'graphs', 'path to directory for storing summaries')

    tf.app.run()