from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

noiseLevel= [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]

dataset_list = []
for i in range(10):
	dataset_list.append(np.load(str(noiseLevel[i])+'.npy'))

dataset5 = dataset_list[4]


plt.gray() # use this line if you don't want to see it in color
plt.imshow(dataset5[53])
plt.show()
