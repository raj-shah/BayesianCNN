from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf

noiseLevel= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

tf.logging.set_verbosity(tf.logging.INFO)

mnist = tf.contrib.learn.datasets.load_dataset("mnist")
# train_data = mnist.train.images # np.array
# train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.train.images # np.array
eval_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = eval_data.reshape(55000,28,28)


for i in range(len(noiseLevel)):
	dataset = np.zeros([55000,28,28])
	count2 = 0
	for image in eval_data:
		image = image.reshape(28,28)
		noisePlaces=np.random.choice([0, 1], size=(28,28), p=[1-noiseLevel[i], noiseLevel[i]])
		noiseScale=np.random.rand(28,28)
		noise=noiseScale*noisePlaces
		noisy_image= np.mod(image + noise, 1)
		#print (noisy_image.shape)
		dataset[count2,:,:] = noisy_image
		count2 += 1
	np.save(str(noiseLevel[i]), dataset)





# plt.gray() # use this line if you don't want to see it in color
# plt.imshow(eval_data[9999])
# plt.show()
# plt.gray() # use this line if you don't want to see it in color
# plt.imshow(dataset[9999])
# plt.show()







