import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from Model import Model
import numpy as np
from get_data import get_data_set

FLAGS = tf.app.flags.FLAGS


def accuracy(logits, y):
	with tf.variable_scope('accuracy') as scope:
		accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1)), dtype=tf.float32), name=scope.name) #calculates how much % of our predictions is correct
		tf.summary.scalar('accuracy', accuracy)
		return accuracy

def evaluate():
	with tf.Graph().as_default():
		model = Model()#keep_prob=FLAGS.prob)

		test_x, test_y, test_l = get_data_set("test")
		x_ = tf.placeholder(tf.float32, shape=[None, 3072], name='x_')
		x = tf.reshape(x_, [-1, 32, 32, 3], name='x') 
		y = tf.placeholder(tf.float32, shape=[None, None], name='y')


		softmax_tensors = tf.placeholder(tf.float32, shape=[None, 100, 10]) #SECOND ARGUMENT IS BATCH SIZE, IDEALLY THIS IS A FLAG
		logits = model._build_net(x)
		softmax = tf.nn.softmax(logits)
		shape = tf.shape(softmax)
		saver = tf.train.Saver(max_to_keep = None)
		mean = tf.reduce_mean(softmax_tensors,0)

		acc = accuracy(mean,y)

		with tf.Session() as sess:
			tf.global_variables_initializer().run()
			saver.restore(sess, FLAGS.checkpoint_file_path)
			accuracy_list = np.zeros(FLAGS.T)
			num_batches=0
			for test_feature_batch, test_label_batch in batch_features_labels(test_x, test_y, batch_size=100):
				num_batches+=1
				softmax_list = []
				batch_accuracy_list=[]
				for i in range(FLAGS.T):
					batch_softmax_i = sess.run([softmax], feed_dict={model.x_: test_feature_batch, model.y: test_label_batch})
					softmax_list.append(batch_softmax_i)
					if i>1:
						arr=np.squeeze(np.array(softmax_list)[:i,:,:])
					else :
						arr=np.array(softmax_list)[0,:,:]
					#accuracy for this batch for first i trials
					accuracy_i = sess.run([acc], feed_dict={softmax_tensors: arr, y: test_label_batch})
					#list from 1-T of accuracy after 1<=i<=T trials
					batch_accuracy_list.append(accuracy_i)
				accuracy_list += batch_accuracy_list
			accuracy_list /= num_batches
			print (accuracy_list)
				#f=open('evals2.out','a')
				#f.write(str(FLAGS.prob)+ ","+str(i+1)+","+str(total_accuracy)+'\n')
				#f.close()
		

def main(argv=None):
    evaluate()

if __name__ == '__main__':
    tf.app.flags.DEFINE_string('checkpoint_file_path', 'checks/model-99000', 'path to checkpoint file')
    tf.app.flags.DEFINE_integer('T', 10, 'Number of Forward Passes')
    tf.app.flags.DEFINE_float('prob', 0.5, 'probability of dropout')
    tf.app.run()
