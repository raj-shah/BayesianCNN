
import tensorflow as tf
from Model import Model
import CIFAR10
import numpy as np
import math

FLAGS = tf.app.flags.FLAGS

def train():
    model = Model()

    with tf.Graph().as_default():

        CIFAR10.data_path = 'data/CIFAR-10/'
        CIFAR10.maybe_download_and_extract()

        class_names = CIFAR10.load_class_names()
        print (class_names)

        #load 50,000 train images (np array)
        images_train, cls_train, labels_train = CIFAR10.load_training_data() #cls is the label, labels_train is one hot encoded
        #load 10,000 test images (np array)
        images_test, cls_test, labels_test = CIFAR10.load_test_data()
        # print (images_test.shape)
        # print (images_test.shape)

        x = tf.placeholder(shape = [None, 32,32,3], dtype = tf.float32, name = 'x')   #mnist dataset is shape 28,28,1
        y = tf.placeholder(shape=[None, 10], dtype=tf.float32, name='y') #10 labels
        keep_prob = tf.placeholder(tf.float32, name='dropout_prob')
        global_step = tf.contrib.framework.get_or_create_global_step()

        logits = model.inference(x, keep_prob=keep_prob)
        loss = model.loss(logits=logits, labels=y)

        accuracy = model.accuracy(logits, y)
        summary_op = tf.summary.merge_all()
        train_op = model.train(loss, global_step=global_step)

        init = tf.global_variables_initializer() 
        saver = tf.train.Saver(max_to_keep = None)

        batch_size = FLAGS.batch_size #batch size, this might not be correct size
        input_size = 50000 #50,000 training images
        porp = int(math.ceil(input_size/batch_size))

        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
            sess.run(init)
            f = open('trainingStdDrop.log', 'a+')
            for i in range(FLAGS.num_iter):
                if i%(porp) == 0:
                    permutation=np.random.permutation(input_size) #create a list with random indexes
                    increment = 0 #restart the increment variable 

                batch_idx = permutation[increment*batch_size:(increment+1)*batch_size]
                increment += 1
                image_batch=images_train[batch_idx] #this is a list with batch size number of elements. Each element is a (32,32,3) array (images)
                label_batch=labels_train[batch_idx] #this is a list with batch size number of elements. Each element is a 10 dimensional vector (1 hot encode)


                _, cur_loss, summary = sess.run([train_op, loss, summary_op],
                                                feed_dict={x: image_batch, y: label_batch, keep_prob: 0.5})
                writer.add_summary(summary, i)
                
                
                if i % 100 == 0:
                    validation_accuracy = accuracy.eval(feed_dict={x: images_test, y: labels_test, keep_prob: 1.0}) 
                    print("Iteration: {}\tLoss: {}\tValidation Accuracy: {}\n".format(i, cur_loss, validation_accuracy))
                    
                    if i % 1000 == 0:
                        f.write('{}, {}, {} \n'.format(i, cur_loss, validation_accuracy))
                        saver.save(sess, FLAGS.checkpoint_file_path+"-"+str(i))
                
            f.close()

def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.flags.DEFINE_integer('batch_size', 128, 'size of training batches')
    tf.app.flags.DEFINE_integer('num_iter', 100000, 'number of training iterations')
    tf.app.flags.DEFINE_string('checkpoint_file_path', 'checkpoints/model.ckpt', 'path to checkpoint file')
    tf.app.flags.DEFINE_string('summary_dir', 'graphs', 'path to directory for storing summaries')

    tf.app.run()
