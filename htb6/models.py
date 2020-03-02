from mxnet import gluon, initializer


def ff(dim, depth):
    net = gluon.nn.Sequential()
    net.add(gluon.nn.Dense(dim, activation='swish'))
    for _ in range(depth):
        net.add(gluon.nn.Dense(dim, activation='swish'))
    net.add(gluon.nn.Dense(1, activation='swish'))
    net.initialize(init=initializer.Normal())
    return net
