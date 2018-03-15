import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from Model import Model
import numpy as np

FLAGS = tf.app.flags.FLAGS



def evaluate():
	with tf.Graph().as_default():

		mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
		labels = mnist.test.labels

		noiseLevel= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
		
		# plt.gray() 
		# plt.imshow(dataset5[53])
		# plt.show()

		x = tf.placeholder(shape=[10000, 28,28,1], dtype=tf.float32, name='x')
		y = tf.placeholder(shape=[10000, 10], dtype=tf.float32, name='y') #10 labels
		softmax_tensors = tf.placeholder(tf.float32, shape=[FLAGS.T, 10000, 10])
	
		model = Model()
		logits = model.inference(x, keep_prob=0.5)
		softmax = tf.nn.softmax(logits)
		shape = tf.shape(softmax)
		saver = tf.train.Saver(max_to_keep = None)

		mean = tf.reduce_mean(softmax_tensors,0)
		accuracy = model.accuracy(mean,y)

		for l in range(len(noiseLevel)):
			data = np.load(str(noiseLevel[l])+'.npy')
			data = data.reshape([10000,28,28,1])
			with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
				tf.global_variables_initializer().run()
				saver.restore(sess, FLAGS.checkpoint_file_path)
				softmax_list = []
				for i in range(FLAGS.T):
					softmaxi = sess.run([softmax],
							            feed_dict={x: data, y: labels})
					softmax_list.append(softmaxi)
			
				mean_softmax = sess.run([mean], feed_dict = {softmax_tensors: np.squeeze(np.array(softmax_list))})
			
			print (np.squeeze(mean_softmax).shape)
			softmax_max = np.max(np.squeeze(mean_softmax), axis = 1)
			print (softmax_max.shape)
			#softmax_correct = np.multiply(np.squeeze(mean_softmax), labels)
			summ = (np.sum(softmax_max))
			print (np.divide(summ, 10000.0))

			
			

def main(argv=None):
    evaluate()

if __name__ == '__main__':
    tf.app.flags.DEFINE_string('checkpoint_file_path', '/home/si318/Desktop/MLSALT4/lenet-all-standard-dropout/checkpoints/model.ckpt-5000000', 'path to checkpoint file')
    tf.app.flags.DEFINE_integer('T', 50, 'number of forward passes')
    tf.app.run()
