import matplotlib.pyplot as plt
import alg


import random


def estimate_profit(prices, alg, start_i):
    money = 20000
    count = 0
    history = []
    value_h = [money for _ in range(start_i)]
    for p in iter(prices[start_i:]):
        history.append(p)
        d = alg(history, count, money)
        if d > 0:
            money -= d*(p+0.25)
        elif d < 0:
            money -= d*p
        count += d
        value_h.append(value(money, count, p))
        if value_h[-1] < 0:
            break
    return value_h[-1]

def normalize(l):
    low = min(l)
    high = max(l)
    for i in range(len(l)):
        l[i] = (l[i]-low)/(high-low)
    return l


def value(money, count, p):
    return money + count*p


with open("optiver_hacktheburgh/market_data.csv") as f:
    lines = f.readlines()
print(f"[S] Read {len(lines)} lines.")

sp = []
esx = []

for l in lines:
    l = l.split(',')
    if len(l) == 0:
        continue
    if l[1] == "ESX-FUTURE":
        esx.append(float(l[2]))
    elif l[1] == "SP-FUTURE":
        sp.append(float(l[2]))
sp_profit = []
esx_profit = []
for i in range(0, len(sp), 100):
  p = estimate_profit(sp, alg.alg, i)
  #sp_profit.append((p-20000)/(len(sp)-i))
  sp_profit.append(p)
  if p <= 0:
    plt.axvline(x=i//100, color='black')
  print(sp_profit[-1])
  #esx_profit.append(estimate_profit(esx, alg.alg, i))
plt.plot(sp_profit, 'r')
plt.axhline(y=sum(sp_profit)/len(sp_profit), color='r')
plt.grid('both')
#plt.plot(esx_profit, 'b')
plt.show()

