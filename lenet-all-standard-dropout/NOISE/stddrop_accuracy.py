import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from Model import Model
import numpy as np

FLAGS = tf.app.flags.FLAGS

def evaluate():
	with tf.Graph().as_default():

		mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
		labels = mnist.test.labels

		noiselevel = [0, 0.2, 0.4, 0.6, 0.8, 1]

		x = tf.placeholder(shape=[10000, 28,28,1], dtype=tf.float32, name='x')
		y = tf.placeholder(shape=[10000, 10], dtype=tf.float32, name='y') #10 labels
		softmax_tensors = tf.placeholder(tf.float32, shape=[FLAGS.T, 10000, 10])
	
		model = Model()
		logits = model.inference(x, keep_prob=1.0)
		softmax = tf.nn.softmax(logits)
		shape = tf.shape(softmax)
		saver = tf.train.Saver(max_to_keep = None)

		mean = tf.reduce_mean(softmax_tensors,0)
		accuracy = model.accuracy(mean,y)

		#this is for each forward pass accuracy
		softmax_each=tf.placeholder(tf.float32, shape=[10000, 10])
		accuracyi=model.accuracy(softmax_each,y)

		
		for j in range(len(noiselevel)):
			data = np.load(str(noiselevel[j])+'.npy')
			data = data.reshape([10000,28,28,1])

			with tf.Session() as sess:
				tf.global_variables_initializer().run()
				saver.restore(sess, FLAGS.checkpoint_file_path)
				softmax_list = []
				accuracy_list = []
				for i in range(FLAGS.T):
					softmaxi = sess.run([softmax],
				                            feed_dict={x: data, y: labels})
					softmax_list.append(softmaxi)

					#added for accuracy of each forward pass:
					#print(softmaxi[0])
					accuracyi1=sess.run([accuracyi],
						feed_dict={softmax_each:np.squeeze(np.array(softmaxi[0])), y: labels})
					accuracy_list.append(accuracyi1)
				
				#mean_prob = sess.run([mean], feed_dict = {softmax_tensors: np.squeeze(np.array(softmax_list))})

				total_accuracy = sess.run([accuracy],
						feed_dict={softmax_tensors: np.squeeze(np.array(softmax_list)), y: labels})
				#print (softmax_list[0].shape)
				standard_deviation=np.std(np.array(accuracy_list))
				f = open('out_stddrop.log', 'a+')
				f.write('noise intensity: {}\n'.format(noiselevel[j]))
				f.write('test accuracy: {}\n'.format(total_accuracy))
				f.close()

			
			

def main(argv=None):
    evaluate()

if __name__ == '__main__':
    tf.app.flags.DEFINE_string('checkpoint_file_path', 'checkpoints/model.ckpt-5000000', 'path to checkpoint file')
    tf.app.flags.DEFINE_integer('T', 2, 'number of forward passes')
    tf.app.run()
