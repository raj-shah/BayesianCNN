import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from Model import Model
import numpy as np
FLAGS = tf.app.flags.FLAGS

#function for MC dropout evaluation
def mc_dropout_logits(x,keep_prob,T):
	model= Model()
	prediction_list=[]
	for i in xrange(T):
		prediction=model.inference(x,keep_prob)
		prediction_list=[prediction_list,prediction]
		print (prediction)
		#logits=tf.reduce_mean(prediction_list,0)
	#return logits
	#print prediction_list

def evaluate():
    with tf.Graph().as_default():
        #images, labels = mnist.load_test_data(FLAGS.test_data)
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        x_ = tf.placeholder(tf.float32, shape=[None, 784]) #data gets loaded as a 28x8 vector
        x = tf.reshape(x_,[-1,28,28,1],name = 'x')   #mnist dataset is shape 28,28,1
        y = tf.placeholder(shape=[None, 10], dtype=tf.float32, name='y') #10 labels

        model = Model()
        softmax_prob_layer=[]
	#logits with drop out
        logits = model.inference(x, keep_prob=0.5)
	#normalized probabilities 
        softmax_prob = tf.nn.softmax(logits)

        #accuracy = model.accuracy(logits, y)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
		
	    #print(sess.run([logits_layer],feed_dict={x_: mnist.test.images, y: mnist.test.labels}))
	    #total_accuracy = sess.run([accuracy],
                                        #feed_dict={x_: mnist.test.images, y: mnist.test.labels})

            saver.restore(sess, FLAGS.checkpoint_file_path)
	    #Monte Carlo stocastic forward pass:
            for i in range(50):
                softmax_prob_layer.append(sess.run([softmax_prob],feed_dict={x_: mnist.test.images, y: mnist.test.labels}))
        
            softmax_prob_sum = tf.convert_to_tensor(np.array(softmax_prob_layer))
            softmax_prob_average = tf.reduce_mean(softmax_prob_sum,0)
            softmax_prob_average = tf.squeeze(softmax_prob_average)
	    #obtain the test accuracy:
            accuracy = model.accuracy(softmax_prob_average, y)
            total_accuracy = sess.run([accuracy],feed_dict={x_: mnist.test.images, y: mnist.test.labels})
            print('Test accuracy: {}'.format(total_accuracy))
	    f = open('trainingAccuracies.log','a+');
	    f.write('Test accuracy: {}'.format(total_accuracy))
	    #print(sess.run([softmax_prob_average],feed_dict={x_: mnist.test.images, y: mnist.test.labels}))
	    #print softmax_prob_average
	    #print softmax_prob
def main(argv=None):
    evaluate()

if __name__ == '__main__':
    tf.app.flags.DEFINE_string('checkpoint_file_path', 'checkpoints/model.ckpt-10000-5000', 'path to checkpoint file')
    tf.app.run()
