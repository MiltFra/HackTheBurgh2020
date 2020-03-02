
def alg(history, amount, money):
    money = money + history[-1]*amount
    if money < 5000:
        limit = 50
        big = 50
        small = 0
    elif money < 20000:
        limit = 100
        big = 100
        small = 0
    elif money < 40000:
        limit = 200
        big = 200
        small = 0
    elif money < 60000:
        limit = 300
        big = 300
        small = 0
    elif money < 80000:
        limit = 400
        big = 400
        small = 0
    else:
        limit = 500
        big = 500
        small = 0
    offset = 0
    LEN = 100
    if len(history) < LEN:
        return 0
    if history[-1] >= max(history[-LEN:-1])*(1+offset):
        d = restrict(min(-big, amount), amount, limit)
        if d != 0:
            return d
    elif history[-1] <= min(history[-LEN:-1])*(1-offset):
        d = restrict(max(big, amount), amount, limit)
        if d != 0:
            return d
    if history[-1] > max(history[-10:-1])*(1+offset):
        d = restrict(-small, amount, limit)
        if d != 0:
            return d
    elif history[-1] < min(history[-10:-1])*(1-offset):
        d = restrict(small, amount, limit)
        if d != 0:
            return d
    avg = last_avg(history, 100)
    lo, hi = minmax_last(history, 100)
    p = history[-1]
    return 0


def restrict(d, amount, limit):
    # d = -d
    if -limit <= d+amount <= limit:
        return d
    else:
        #print("Restricting order")
        return 0


def last_avg(history, n):
    if len(history) < n+1:
        return history[-1]
    return sum(history[-n-1:-1]) / len(history)


def minmax_last(values, n):
    if len(values) < n:
        return min(values[:-1]), max(values[:-1])
    return min(values[-n-1:-1]), max(values[-n-1:-1])
