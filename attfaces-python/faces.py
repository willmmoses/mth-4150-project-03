import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt
from math import floor, ceil

def loadface(imagedir,subject,pose):
  """
  Load in the face for the given subject number (integer) and the given
  pose (integer). Directory of images is passed in as imagedir. 
  """
  filename =  f"{imagedir}/s{subject}/{pose}.pgm"
  X = image.imread(filename).astype(np.double);  # read, convert to double precision
  return X.flatten()

def showfaces(X):
  """
  Accepts a matrix of image vectors (assumed to be from 112 x 92 images, and
  with the image vectors as columns) and plots them in a grid. 
  Will plot at most 16 faces.
  """
  n = X.shape[1]
  if n > 16:
    raise ValueError('A maximum of 16 faces please!')
  rows = ceil(n/4);
  cols = 4;

  fig = plt.figure(figsize=(10,10))
  for j in range(n):  
    pic = X[:,j].reshape(112,92)
    ax = fig.add_subplot(rows,cols,j+1)
    ax.axis('off')
    ax.imshow(pic, cmap='gray')