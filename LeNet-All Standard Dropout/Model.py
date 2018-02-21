import tensorflow as tf
import poly_inverse_time_decay as td


class Model(object):
    def __init__(self, batch_size=64, learning_rate=0.01, num_labels=10):
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._num_labels = num_labels

    def inference(self, images, keep_prob):
        with tf.variable_scope('conv1') as scope:
            kernel = self._create_weights([5, 5, 1, 20]) #[filter_height, filter_width, in_channels, out_channels]
            conv = self._create_conv2d(images, kernel) #create the conv layer
            bias = self._create_bias([20]) #create bias variable
            preactivation = tf.nn.bias_add(conv, bias) #add bias
            conv1 = tf.nn.relu(preactivation, name=scope.name) #put through RELU activation function. #output size [batch_size, 28,28,20]
            self._activation_summary(conv1) #this is for TensorBoard visualization

        dropout1 = tf.nn.dropout(conv1, keep_prob) #do dropout
        h_pool1 = self._create_max_pool_2x2(dropout1) #do max pooling. output size [batch_size, 14,14,20]

        with tf.variable_scope('conv2') as scope:
            kernel = self._create_weights([5, 5, 20, 50])
            conv = self._create_conv2d(h_pool1, kernel)
            bias = self._create_bias([50])
            preactivation = tf.nn.bias_add(conv, bias)
            conv2 = tf.nn.relu(preactivation, name=scope.name) #outputsize [batch_size, 14,14,50]
            self._activation_summary(conv2)


        dropout2 = tf.nn.dropout(conv2, keep_prob)
        h_pool2 = self._create_max_pool_2x2(dropout2) #output size [batch_size, 7, 7, 50]

        with tf.variable_scope('dense') as scope:
            reshape = tf.reshape(h_pool2, [-1, 7 * 7 * 50])
            W_dense = self._create_weights([7 * 7 * 50, 500])
            b_dense = self._create_bias([500])
            dense = tf.nn.relu(tf.matmul(reshape, W_dense) + b_dense, name=scope.name)
            self._activation_summary(dense)

        with tf.variable_scope('logit') as scope:
            W_logit = self._create_weights([500, self._num_labels])
            b_logit = self._create_bias([self._num_labels])
            dense_drop = tf.nn.dropout(dense, keep_prob)
            logit = tf.nn.bias_add(tf.matmul(dense_drop, W_logit), b_logit, name=scope.name)
            self._activation_summary(logit)
        return logit

    def train(self, loss, global_step):
        #should probably make these variables arguements but cba
        decay_step = 1
        decay_rate = 0.0001 #decay rate
        #learning_rate = tf.train.inverse_time_decay(self._learning_rate, global_step, decay_step, decay_rate)
        learning_rate = td.poly_inverse_time_decay(self._learning_rate, global_step, decay_steps = 1, decay_rate =  decay_rate, power = 0.75)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum = 0.9)#, use_nesterov=True)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return train_op

    def loss(self, logits, labels):
        with tf.variable_scope('loss') as scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            cost = tf.reduce_mean(cross_entropy, name=scope.name) #computes the mean loss = cost. Minimizing mean loss.
            tf.summary.scalar('cost', cost)

        return cost

    def accuracy(self, logits, y):
        with tf.variable_scope('accuracy') as scope:
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1)), dtype=tf.float32),
                                      name=scope.name) #calculates how much % of our predictions is correct
            tf.summary.scalar('accuracy', accuracy)
        return accuracy

    def _create_conv2d(self, x, W):
        return tf.nn.conv2d(input=x,
                            filter=W, #4d tensor that includes the size of the filter and number of filters
                            strides=[1, 1, 1, 1], #The stride of the sliding window for each dimension of the input tensor.
                            padding='SAME') #padding set so the output feature maps are the same size as the input feature maps.

    def _create_max_pool_2x2(self, input):
        return tf.nn.max_pool(value=input,
                              ksize=[1, 2, 2, 1], #The size of the window for each dimension of the input tensor.
                              strides=[1, 2, 2, 1], #The stride of the sliding window for each dimension of the input tensor.
                              padding='SAME') # padding set so the output feature maps are the same size as the input feature maps.

    def _create_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1, dtype=tf.float32))

    def _create_bias(self, shape):
        return tf.Variable(tf.constant(1., shape=shape, dtype=tf.float32))

    def _activation_summary(self, x):
        #This is simply to catch information for visualization using tensorboard
        tensor_name = x.op.name
        tf.summary.histogram(tensor_name + '/activations', x) 
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
