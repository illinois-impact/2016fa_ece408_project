import numpy as np
import h5py
# import scipy
# from scipy import misc


def main():
    # Load data.
    f = h5py.File('data.hdf5', 'r')
    X = f['x'][:]
    Y = f['y'][:]
    f.close()

    # Load model. Expected to get ~97% accuracy on the provided data.
    f = h5py.File('model.hdf5', 'r')
    conv1 = f['conv1'][:]
    conv2 = f['conv2'][:]
    fc1 = f['fc1'][:]
    fc2 = f['fc2'][:]
    f.close()

    num_correct = 0
    num_total = 0
        # Show the first two images.
        # scipy.misc.imshow(X[0:1,:,:,:].squeeze())  # 7
        # scipy.misc.imshow(X[1:2,:,:,:].squeeze())  # 2
    print forward_operation(X, conv1, conv2, fc1, fc2)


def forward_operation(X, conv1, conv2, fc1, fc2):
    '''Forward operation for the CNN, a combination of
      conv layer + average pooling + relu.'''

    # X:  batch_size x 28 x 28 x 1
    net = conv_forward_valid(X, conv1)  # batch_size 24 x 24 x 32
    net = relu(net)  # batch_size 24 x 24 x 32
    net = average_pool(net)  # batch_size 12 x 12 x 32
    net = conv_forward_valid(net, conv2)  # batch_size 8 x 8 x 64
    net = relu(net)  # batch_size 8 x 8 x 64
    net = average_pool(net)  # batch_size x 4 x 4 x 64
    net = np.reshape(net, (net.shape[0], -1))  # batch_size x 1024
    net = fully_forwrad(net, fc1)  # batch_size x 128
    net = relu(net)  # batch_size x 128
    net = fully_forwrad(net, fc2)  # batch_size x 10

    return np.argmax(net, 1)  # batch_size x 1


def conv_forward_valid(X, W):
    '''Implemented following the notes in Figure 16.4'''

    # Unnamed variable is num_channels
    batch_size, num_rows, num_cols, _ = X.shape
    filter_h, filter_w, in_channel, out_channel = W.shape
    Y = np.zeros((batch_size, num_rows - filter_h + 1,
                  num_cols - filter_w + 1, out_channel))
    # Expand out the conv layer operation.
    for i in range(0, batch_size):
        for m in range(0, out_channel):
            for h in range(0, Y.shape[1]):
                for w in range(0, Y.shape[2]):
                    for c in range(0, in_channel):
                        for p in range(0, filter_h):
                            for q in range(0, filter_w):
                                Y[i, h, w, m] += X[i, h + p,
                                                   w + q, c] * W[p, q, c, m]
    return Y


def relu(X):
    '''Apply recified linear unit to X'''
    X[X < 0] = 0
    return X


def average_pool(X, pool_size=2):
    ''' Implemented following the notes in Figure 16.5'''
    batch_size, H, W, M = X.shape  # batch_size, height, width, channels.
    Y = np.zeros((batch_size, H / pool_size, W / pool_size, M))
    for i in range(0, batch_size):
        for m in range(0, M):
            for h in range(0, H / pool_size):
                for w in range(0, W / pool_size):
                    for p in range(0, pool_size):
                        for q in range(0, pool_size):
                            Y[i, h, w, m] += X[i, pool_size * h + p,
                                               pool_size * w + q, m] / (1.0 * pool_size**2)
    return Y


def fully_forwrad(X, W):
    '''matrix multiplication...'''
    return np.dot(X, W)

if __name__ == '__main__':
    main()
