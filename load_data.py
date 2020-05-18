import matplotlib.pyplot as plt
import numpy as np
import sys


def to_data(lines):
    data = np.zeros(len(lines), dtype="float")
    for i, l in enumerate(lines):
        data[i] = np.array(float(l[1]))
    return data


def normalize_data(data):
    avg = np.average(data)
    analysis = SingleAnalysis(data)
    res = np.zeros(len(data))
    for i, p in enumerate(data):
        res[i] = p/avg
    return res


def to_interval_averages(data, n):
    res = np.zeros(len(data)//n)
    for i in range(len(data)//n):
        res[i] = np.average(data[i*n:(i+1)*n])
    return res


class SingleAnalysis:
    def __init__(self, data):
        self.data = data
        self.min = sys.float_info.max
        self.max = sys.float_info.min
        for p in data:
            if p > self.max:
                self.max = p
            elif p < self.min:
                self.min = p

    def __str__(self):
        return f"Min: {self.min}, Max: {self.max}"


class DoubleAnalysis:
    def __init__(self, data1, data2):
        self.data1, self.data2 = data1, data2
        self.min_delta = sys.float_info.max
        self.max_delta = sys.float_info.min
        for p1, p2 in zip(data1, data2):
            if p1-p2 < self.min_delta:
                self.min_delta = p1-p2
            elif p1 - p2 > self.max_delta:
                self.max_delta = p1-p2

    def __str__(self):
        return f"Min: {self.min_delta}, Max: {self.max_delta}"


lines = []

with open("optiver_hacktheburgh/market_data.csv") as f:
    lines = f.readlines()[5000:6000]
print(f"[S] Read {len(lines)} lines.")

sp = []
esx = []

for l in lines:
    l = l.split(',')
    if len(l) == 0:
        continue
    if l[1] == "ESX-FUTURE":
        esx.append(l[1:])
    elif l[1] == "SP-FUTURE":
        sp.append(l[1:])

with open("optiver_hacktheburgh/trades.csv") as f:
    lines = f.readlines()[5000:6000]

sp_i = 0
sp_sells = []
sp_buys = []
sp_volumes = []
esx_i = 0
esx_buys = []
esx_sells = []
esx_volumes = []
esx_cols = []
for l in lines:
    l = l.split(',')
    if len(l) == 0:
        continue
    if l[1] == "ESX-FUTURE":
        if l[2] == "BID":
            esx_buys.append((esx_i, l[3]))
            esx_cols.append("red")
        elif l[2] == "ASK":
            esx_sells.append((esx_i, l[3]))
            esx_cols.append("green")
        esx_i += 1
        esx_volumes.append(float(l[4]))
    elif l[1] == "SP-FUTURE":
        if l[2] == "BID":
            sp_buys.append((sp_i, l[3]))
        else:
            sp_sells.append((sp_i, l[3]))
        sp_i += 1
        sp_volumes.append(float(l[4]))

sp_data = to_data(sp)
esx_data = to_data(esx)
sp_buy_data = to_data(sp_buys)
sp_sell_data = to_data(sp_sells)
esx_buy_data = to_data(esx_buys)
esx_sell_data = to_data(esx_sells)

sp_norm = normalize_data(sp_data)
esx_norm = normalize_data(esx_data)
N = 500


#plt.plot(sp_norm, 'r')
#plt.plot(esx_norm, 'g')
plt.plot(sp_data, 'r')
plt.scatter(range(len(sp_data)), sp_data, c=esx_cols, s=[x // 20 for x in esx_volumes])
#indices = []
#for i, _ in sp_sells:
#    indices.append(i)
#plt.plot(indices, sp_sell_data, 'r.')
#indices = []
#for i, _ in sp_buys:
#    indices.append(i)
#plt.plot(indices, sp_buy_data, 'g.')
#plt.plot(esx_data, 'g')
#indices = []
#for i, _ in esx_sells:
#    indices.append(i)
#plt.scatter(indices, normalize_data(esx_sell_data), s=[esx_volumes[i]/50 for i in indices])
#indices = []
#for i, _ in esx_buys:
#    indices.append(i)
#plt.scatter(indices, normalize_data(esx_buy_data), s=[esx_volumes[i]/50 for i in indices])
#plt.bar(range(len(esx_volumes)), esx_volumes)
plt.show()
