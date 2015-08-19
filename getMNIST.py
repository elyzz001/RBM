import urllib
import gzip
import struct
from array import array
import numpy as np



"""
	Script adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
	and python documentation
"""


train_img_url="http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
train_label_url="http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"

test_img_url="http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
test_label_url="http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"


train_img_gz="train-images-idx3-ubyte.gz"
train_label_gz="train-labels-idx1-ubyte.gz"

trainIm_file="train-images-idx3-ubyte"
trainLabel_file="train-labels-idx1-ubyte"

print "Downloading MNIST from http://yann.lecun.com/exdb/mnist/index.html ..."

print "Downloading training set...",
urllib.urlretrieve(train_img_url,train_img_gz)
urllib.urlcleanup()
urllib.urlretrieve(train_label_url,train_label_gz)
urllib.urlcleanup()
print "Done!"


print "Extracting ubyte files from gz files..."
print "-"+train_img_gz
print "-"+train_label_gz

f_in=gzip.open(train_img_gz,'rb')
f_out=open(trainIm_file,'wb')
f_out.write(f_in.read())
f_in.close()
f_out.close()

f_in=gzip.open(train_label_gz,'rb')
f_out=open(trainLabel_file,'wb')
f_out.write(f_in.read())
f_in.close()
f_out.close()

print "...Done!"

print "Extracting training data from ubyte file into numpy arrays...",

flbl = open(trainLabel_file, 'rb')
magic_nr, size = struct.unpack(">II", flbl.read(8))
lbl = array("b", flbl.read())
flbl.close()

fimg = open(trainIm_file, 'rb')
magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
img=array("B",fimg.read())
fimg.close()

ind = [ k for k in xrange(size)]
N=len(ind)
trainIm =  np.zeros((rows,cols,N),dtype=np.uint8)
trainLabel = np.zeros((N, 1),dtype=np.uint8)



for i in xrange(N):
    trainIm[:, :,i] = np.array(img[ i*rows*cols : (i+1)*rows*cols]).reshape(rows,cols)
    trainLabel[i] = lbl[i]



print "Done!"
print "Dumping numpy arrays into pickles!!!!!....",
trainIm.dump("trainIm.pkl");
trainLabel.dump("trainLabel.pkl")
print "Done!"

