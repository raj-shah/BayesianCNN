import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from Model import Model

FLAGS = tf.app.flags.FLAGS


def evaluate():
    with tf.Graph().as_default():
        #images, labels = mnist.load_test_data(FLAGS.test_data)
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        x_ = tf.placeholder(tf.float32, shape=[None, 784]) #data gets loaded as a 28x8 vector
        x = tf.reshape(x_,[-1,28,28,1],name = 'x')   #mnist dataset is shape 28,28,1
        y = tf.placeholder(shape=[None, 10], dtype=tf.float32, name='y') #10 labels

        model = Model()
        logits = model.inference(x, keep_prob=1.0)
        accuracy = model.accuracy(logits, y)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver.restore(sess, FLAGS.checkpoint_file_path)

            total_accuracy = sess.run([accuracy],
                                        feed_dict={x_: mnist.test.images, y: mnist.test.labels})
            print('Test accuracy: {}'.format(total_accuracy))


def main(argv=None):
    evaluate()

if __name__ == '__main__':
    tf.app.flags.DEFINE_string('checkpoint_file_path', 'checkpoints/model.ckpt-10000-10000', 'path to checkpoint file')
    tf.app.run()