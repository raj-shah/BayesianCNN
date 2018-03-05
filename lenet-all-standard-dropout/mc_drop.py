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
		softmax_tensors = tf.placeholder(tf.float32, shape=[FLAGS.T, 10000, 10])
	
		model = Model()
		logits = model.inference(x, keep_prob=0.5)
		softmax = tf.nn.softmax(logits)
		shape = tf.shape(softmax)
		saver = tf.train.Saver(max_to_keep = None)

		mean = tf.reduce_mean(softmax_tensors,0)
		accuracy = model.accuracy(mean,y)


		#this is for each forward pass accuracy
		softmax_each=tf.placeholder(tf.float32, shape=[10000, 10])
		accuracyi=model.accuracy(softmax_each,y)


		softmax_list = []
		accuracy_list = []
		with tf.Session() as sess:
			tf.global_variables_initializer().run()
			saver.restore(sess, FLAGS.checkpoint_file_path)
			for i in range(FLAGS.T):
				softmaxi = sess.run([softmax],
				                            feed_dict={x_: mnist.test.images, y: mnist.test.labels})
				softmax_list.append(softmaxi)



				#added for accuracy of each forward pass:
				print(softmaxi[0])
				accuracyi1=sess.run([accuracyi],
						feed_dict={softmax_each:np.squeeze(np.array(softmaxi[0])), y: mnist.test.labels})
				accuracy_list.append(accuracyi1)
				
				
			#mean_prob = sess.run([mean], feed_dict = {softmax_tensors: np.squeeze(np.array(softmax_list))})

			total_accuracy = sess.run([accuracy],
						feed_dict={softmax_tensors: np.squeeze(np.array(softmax_list)), y: mnist.test.labels})
			#print (softmax_list[0].shape)
			standard_deviation=np.std(np.array(accuracy_list))

			print('Test accuracy: {}'.format(total_accuracy))
			print('Standard deviation of 10 forward passes:',standard_deviation)
			print('Accuracy list is',accuracy_list)

			
			

def main(argv=None):
    evaluate()

if __name__ == '__main__':
    tf.app.flags.DEFINE_string('checkpoint_file_path', 'checkpoints/model.ckpt-180000', 'path to checkpoint file')
    tf.app.flags.DEFINE_integer('T', 10, 'Number of Forward Passes')
    tf.app.run()
