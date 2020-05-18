import numpy as np
from mxnet import nd, gluon, initializer, autograd
from time import time
#import models
#from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.trainer import Trainer
from matplotlib import pyplot as plt
from gluonts.evaluation.backtest import make_evaluation_predictions


def acc(output, label):
    return (
        output.argmax(axis=1) == label.argmax(
            axis=1
        )
    ).mean().asscalar()


def easy_train():
    import pandas as pd
    df = pd.read_csv("optiver_hacktheburgh/sp.csv",
                     header=0, index_col=0, usecols=[0, 2], skiprows=lambda x: x % 5 != 0)
    # df[:100].plot(linewidth=2)
    print("Showing")
    # plt.show()
    from gluonts.dataset.common import ListDataset
    training_data = ListDataset(
        [{"start": df.index[0], "target": df.values.flatten()}],
        freq="1s"
    )
    #from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
    from gluonts.model.deepar import DeepAREstimator
    from gluonts.trainer import Trainer
    estimator = DeepAREstimator(
        freq="1min", prediction_length=100, trainer=Trainer(epochs=20))
    predictor = estimator.train(training_data=training_data)
    test_data = ListDataset(
        [{"start": df.index[0], "target": df.values.flatten()[:1000]}],
        freq="10s"
    )
    full_test_data = ListDataset(
        [{"start": df.index[0], "target": df.values.flatten()}],
        freq="10s"
    )

    means = []
    for i, (test_entry, forecast) in enumerate(zip(full_test_data, predictor.predict(test_data))):
        # if i > 0:
        #  break
        print(forecast.dim())
        plt.plot(test_entry["target"])
        #forecast.plot(color='g', prediction_intervals=[], output_file="test.png")
        means.extend(list(forecast.mean))
        print(forecast.mean)
    l = len(test_entry["target"])
    plt.axhline(y=means[0],xmin=0, xmax=l, linewidth=2, color='r')
    plt.axvline(x=5000, color='b')
    plt.grid(which='both')
    plt.show()


def train(lr=1, batch_size=10, min_change=5, patience=5, dim=10, step=10):
    # Data
    train_data, valid_data = get_data(dim, step, 0)  # (step, 2, ex_l, dim+1)
    print("[S] Obtained data.")
    loss_f = gluon.loss.KLDivLoss(from_logits=False)
    net = models.ff(dim, 3)
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'learning_rate': lr}
    )
    # Variables
    best_loss = 0
    train_acc = 0
    valid_acc = 0
    avg_train_acc = 0
    avg_valid_acc = 0
    epoch = 0
    best_loss = float('inf')
    last_decrease = 0
    stats = []
    # Training
    print("[S] Starting the training.")
    for epoch in range(20):
        train_loss, train_acc, valid_acc = 0., 0., 0.
        tic = time()
        for data_sets, label in train_data:
            print(data_sets)
            # forward + backward
            with autograd.record():
                output = net(data_sets)
                loss = loss_f(output, label)
            loss.backward()
            # update parameters
            trainer.step(batch_size)
            # calculate training metrics
            train_loss += loss.mean().asscalar()
            train_acc += acc(output, label)


def get_data(dim, step, look_ahead):
    with open("optiver_hacktheburgh/market_data.csv") as f:
        lines = f.readlines()
    sp = []
    esx = []
    for l in lines:
        l = l.split(',')
        if len(l) == 0:
            continue
        if l[1] == "ESX-FUTURE":
            esx.append(l[2:])
        elif l[1] == "SP-FUTURE":
            sp.append(l[2:])
    data = to_data(sp)
    avgs = to_averages(data, 0, step)
    ex_l = len(avgs)-dim-look_ahead-3
    X = np.zeros((step, ex_l, dim))
    y = np.zeros((step, ex_l))
    for i in range(step):
        sX, sy = offset_sample(dim, look_ahead, step, i, data)
        X[i] = sX[:ex_l]
        y[i] = sy[:ex_l]
    return gluon.data.dataset.ArrayDataset(X[:step//2], y[:step//2]), gluon.data.dataset.ArrayDataset(X[step//2:], y[step//2:])


def offset_sample(dim, look_ahead, step, offset, data):
    avgs = to_averages(data, offset, step)
    l = len(avgs) - dim - look_ahead-1
    X = np.zeros((l, dim))
    y = np.zeros(l)
    for i in range(l):
        p = np.zeros(dim)
        p[:dim] = avgs[i:i+dim]
        p = p / p[0]
        X[i] = p
        y[i] = avgs[i+dim+look_ahead+1] / p[0]
    return X, y


def to_data(lines):
    data = np.zeros(len(lines), dtype="float")
    for i, l in enumerate(lines):
        data[i] = float(l[0])
    return data


def to_averages(data, offset, step):
    if offset > len(data):
        return None
    data = data[offset:]
    l = (len(data))//step
    res = np.zeros(l)
    for i in range(l):
        res[i] = np.average(data[i*step:(i+1)*step])
    return res


if __name__ == "__main__":
    easy_train()
