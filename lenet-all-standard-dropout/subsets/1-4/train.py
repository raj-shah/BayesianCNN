
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from Model import Model
import math
import numpy as np

FLAGS = tf.app.flags.FLAGS

def train():
    model = Model()

    with tf.Graph().as_default():
        # Load training data and use test as evaluation set in one hot format
        mnist_ = input_data.read_data_sets("MNIST_data/", one_hot=True)
        perm =np.random.permutation(13750)
        images = mnist_.train.images[perm]
        labels = mnist_.train.labels[perm]
        print (images.shape) #[13750,28,28,1] np array

        x_ = tf.placeholder(tf.float32, shape=[None, 784]) #data gets loaded as a 28x8 vector
        x = tf.reshape(x_,[-1,28,28,1],name = 'x')   #mnist dataset is shape 28,28,1
        y = tf.placeholder(shape=[None, 10], dtype=tf.float32, name='y') #10 labels
        keep_prob = tf.placeholder(tf.float32, name='dropout_prob')
        global_step = tf.contrib.framework.get_or_create_global_step()

        logits = model.inference(x, keep_prob=keep_prob)
        loss = model.loss(logits=logits, labels=y)

        accuracy = model.accuracy(logits, y)
        summary_op = tf.summary.merge_all()
        train_op = model.train(loss, global_step=global_step)

        batch_size = FLAGS.batch_size #batch size, this might not be correct size
        input_size = 13750 
        porp = int(math.ceil(input_size/batch_size))

        init = tf.global_variables_initializer() 
        saver = tf.train.Saver(max_to_keep = 100000)

        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
            sess.run(init)
            for i in range(FLAGS.num_iter):
                if i%(porp) == 0:
                    permutation=np.random.permutation(input_size) #create a list with random indexes
                    increment = 0 #restart the increment variable 

                batch_idx = permutation[increment*batch_size:(increment+1)*batch_size]
                increment += 1
                image_batch=images[batch_idx] #this is a list with batch size number of elements. Each element is a (32,32,3) array (images)
                label_batch=labels[batch_idx]

                _, cur_loss, summary = sess.run([train_op, loss, summary_op],
                                                feed_dict={x_: image_batch, y: label_batch, keep_prob: 0.5})

                writer.add_summary(summary, i)
                if i % 5000 == 0:
                    f = open('trainingStdDrop.log', 'a+')
                    validation_accuracy = accuracy.eval(feed_dict={x_: mnist_.test.images, y: mnist_.test.labels, keep_prob: 1.0}) 
                    f.write('{}, {}, {} \n'.format(i, cur_loss, validation_accuracy))
                    f.close()
                    saver.save(sess, FLAGS.checkpoint_file_path+"-"+str(i))

def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.flags.DEFINE_integer('batch_size', 64, 'size of training batches')
    tf.app.flags.DEFINE_integer('num_iter', 1000000, 'number of training iterations')
    tf.app.flags.DEFINE_string('checkpoint_file_path', 'checkpoints/model.ckpt', 'path to checkpoint file')
    tf.app.flags.DEFINE_string('summary_dir', 'graphs', 'path to directory for storing summaries')

    tf.app.run()
