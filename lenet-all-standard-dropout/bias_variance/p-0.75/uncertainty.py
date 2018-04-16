import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from Model import Model
import numpy as np
FLAGS = tf.app.flags.FLAGS
def evaluate():
    with tf.Graph().as_default():
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        x_ = tf.placeholder(tf.float32, shape=[None, 784]) #data gets loaded as a 28x8 vector
        x = tf.reshape(x_,[-1,28,28,1],name = 'x')   #mnist dataset is shape 28,28,1
        y = tf.placeholder(shape=[None, 10], dtype=tf.float32, name='y') #10 labels
        softmax_tensors = tf.placeholder(tf.float32, shape=[None, 10000, 10])
    
        model = Model()
        logits = model.inference(x, keep_prob=FLAGS.prob)
        softmax = tf.nn.softmax(logits)
        shape = tf.shape(softmax)
        saver = tf.train.Saver(max_to_keep = None)
        mean = tf.reduce_mean(softmax_tensors,0)
        accuracy = model.accuracy(mean,y)
        softmax_list = []
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver.restore(sess, FLAGS.checkpoint_file_path)
            for i in range(FLAGS.T):
                softmaxi = sess.run([softmax],
                                            feed_dict={x_: mnist.test.images, y: mnist.test.labels}) 
                softmax_list.append(softmaxi)
            for i in range(FLAGS.T):
                if i>1:
                    arr=np.squeeze(np.array(softmax_list)[:i,:,:])
                else :
                    arr=np.array(softmax_list)[0,:,:]
                total_accuracy, soft = sess.run([accuracy,mean],
                        feed_dict={softmax_tensors: arr, y: mnist.test.labels})
                f=open('bias_var.log','a')
                f.write(str(FLAGS.prob)+ ","+str(i+1)+","+str(total_accuracy)+'\n')
                f.close()
                #L=np.array([np.squeeze(np.array(soft)).flatten(), mnist.test.labels.flatten()])
                #L=np.transpose(L)
                #L=L[L[:,0].argsort()]
                #np.savetxt(str(i)+'uncertainty2.out', L, delimiter=',')
        
            
            
def main(argv=None):
    evaluate()
if __name__ == '__main__':
    tf.app.flags.DEFINE_string('checkpoint_file_path', 'checkpoints/model.ckpt-240000', 'path to checkpoint file')
    tf.app.flags.DEFINE_integer('T', 2, 'Number of Forward Passes')
    tf.app.flags.DEFINE_float('prob', 1.0, 'probability of dropout')
    tf.app.run()
