import urllib
import gzip


train_img_url="http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
train_label_url="http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"

test_img_url="http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
test_label_url="http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"


train_img_gz="train-images-idx3-ubyte.gz"
train_label_gz="train-labels-idx1-ubyte.gz"



print "Downloading MNIST from http://yann.lecun.com/exdb/mnist/index.html ..."
print "Downloading training set...",
urllib.urlretrieve(train_img_url,train_label_gz)
urllib.urlcleanup()
print "Done!"
print "Extracting images from gz files...",
print train_img_gz
print train_label_gz
print "...Done!"


