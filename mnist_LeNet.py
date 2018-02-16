from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def LeNet_all(features, labels, mode):
#This function defines our CNN structure.

  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1]) #batchsize, width, height, channels.
  #reshaping the data to expected shape. -1 just means that we define batchsize as how many inputs we have. (So not defined)

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer, #define the input
      filters=20, #number of filters
      kernel_size=[5, 5], #size of the filter/kernel
      padding="same", #padding set so the output feature maps are the same size as the input feature maps.
      activation=tf.nn.relu) #Relu activation function
  	  #output size will be [batch_size, 28,28,20] (since we had 20 filter
  dropout1 = tf.layers.dropout(
	  inputs=conv1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN) #dropout layer
  pool1 = tf.layers.max_pooling2d(inputs=dropout1, pool_size=[2, 2], strides=2) #maxpool layer
  #output size now [batch size, 14,14,20] after pooling

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=50,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  	  #output size: [batch size, 14,14,50]
  dropout2 = tf.layers.dropout(
	  inputs=conv2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
  pool2 = tf.layers.max_pooling2d(inputs=dropout2, pool_size=[2, 2], strides=2)
  #output size [7,7,50]

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 50])
  dense = tf.layers.dense(inputs=pool2_flat, units=500, activation=tf.nn.relu) #this is our fully connected layer
  dropout3 = tf.layers.dropout(
      inputs=dense, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN) #droput.  rate = dropout rate. 
  	  #training is a boolean to specify whether we are training or not. Dropout is only performed when training = True.

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout3, units=10) #This is our output layer/unscaled probabilities. takes in dropout ([batchsize, 500]) and outputs
  # [batch_size, 10].


  #create dictionary with our predictions, and then return it as an EstimatorSpec with appropriate information
  predictions = {
      "classes": tf.argmax(input=logits, axis=1), #class with highest value gets picked. 
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor") #probability of each class, using softmax
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  #define our loss function. Cross entropy in this case.
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
  	global_step = tf.Variable(0, trainable=False) #Passing global_step to minimize() will increment it at each step
  	learning_rate = 0.01
  	decay_step = 1
  	decay_rate = 0.0001 #decay rate
  	learning_rate = tf.train.inverse_time_decay(learning_rate, global_step, decay_step, decay_rate)
  	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  	train_op = optimizer.minimize(
  		loss=loss,
  		global_step=tf.train.get_global_step())
  	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

#Now having defined the model

def main(unused_argv):
	  # Load training and eval data
	#--------------------------------------------------------
	mnist = tf.contrib.learn.datasets.load_dataset("mnist")
	train_data = mnist.train.images # np.array
	train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
	eval_data = mnist.test.images # np.array
	eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
	#--------------------------------------------------------

	#Create estimator:
	mnist_classifier = tf.estimator.Estimator(
	model_fn=LeNet_all, model_dir="mnist_convnet_model/")
	# Set up logging for predictions
	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
	  tensors=tensors_to_log, every_n_iter=50)
	#train the model
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
	    x={"x": train_data}, #training features
	    y=train_labels, #training labels
	    batch_size=64, 
	    num_epochs=None, #model will train until the specified number of steps is reached.
	    shuffle=True) #shuffle the data
	mnist_classifier.train(
	    input_fn=train_input_fn,
	    steps=1000000, #model will train for 20000 steps
	    hooks=[logging_hook]) #specify the logging hook.
	# Evaluate the model and print results
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
	    x={"x": eval_data},
	    y=eval_labels,
	    num_epochs=1, #since we only do 1 forward run. Change this is MC dropout I think.
	    shuffle=False)
	eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
	print(eval_results)

if __name__ == "__main__":
  tf.app.run()





















