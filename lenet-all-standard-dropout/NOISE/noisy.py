from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

noiseLevel= [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]

tf.logging.set_verbosity(tf.logging.INFO)

mnist = tf.contrib.learn.datasets.load_dataset("mnist")
# train_data = mnist.train.images # np.array
# train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images # np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
eval_data = eval_data.reshape(10000,28,28)




for i in range(len(noiseLevel)):
	dataset = np.zeros([10000,28,28])
	count2 = 0
	for image in eval_data:
		image = image.reshape(28,28)
		noisePlaces=np.random.choice([0, 1], size=(28,28), p=[1-noiseLevel[i], noiseLevel[i]])
		noiseScale=np.random.rand(28,28)
		noise=noiseScale*noisePlaces
		noisy_image= np.mod(image + noise, 2)
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







