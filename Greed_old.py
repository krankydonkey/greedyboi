from math import factorial


def binomial(x, n):
    return factorial(n) / (factorial(x) * factorial(n - x))


def probability(x, n, s):
    return (1 / s) ** x * ((s - 1) / s) ** (n - x)


def prob_in(x, n, s=6):
    return binomial(x, n) * probability(x, n, s)


def greater_in(x, n, s=6):
    return sum([prob_in(i, n, s) for i in range(x, n+1)])


def greater_in2(x, n):
    return binomial(x, n) * 6**(n-x) * (1 / 6) ** n


def dg(x, n):
    prob = 0
    for i in range(0, n+1):
        if i >= x:
            # Roll equal to or more than the required number of dice
            prob += prob_in(i, n)
        elif i > 0:
            # Roll less than the required number of dice
            prob += prob_in(i, n) * dg(x-i, n-i)
        elif n > x:
            # Roll none of the required dice, but one of the other kind and keep scoring
            prob += prob_in(1, n) * dg(x, n - 1)
    return prob


def six():
    return prob_in(6, 6) * 4 + (greater_in(3, 6) - prob_in(6, 6)) * prob_in(3, 3) * 4 + dg(6, 6) * 2


def odds(results, n, s=6):
    total = sum(results)
    if total == 0:
        return 0
    num = 1
    left = n
    for x in results:
        num *= factorial(left) / (factorial(x) * factorial(left - x))
        left -= x
    return num * (1 / s) ** total * ((s - 1) / s) ** (n - total)


def greater_odds(other, greater, n, s=6):
    if not greater:
        return odds(other, n, s)
    left = n - sum(other) - sum(greater)
    return sum([greater_odds(other + [greater[0] + i], greater[1:], n, s) for i in range(left+1)])


def dg2(x, n):
    prob = 0
    for i in range(0, n + 1):
        if i >= x:
            # Roll equal to or more than the required number of dice
            prob += odds([i], n)
        elif i > 0:
            # Roll less than the required number of dice
            prob += odds([i], n) * dg(x - i, n - i)
        elif n > x:
            # Roll none of the required dice, but one of the other kind and keep scoring
            prob += odds([1], n) * dg(x, n - 1)
    return prob


class Greed:
    sides = 6
    num_dice = 6
    all_same = 5000
    one_each = 1000
    groups = ((3, 600), (3, 500), (3, 400), (3, 300), (3, 300), (4, 1000))
    solo = (0, 50, 0, 0, 0, 100)
    goal = 5000

    def __init__(self):
        self.dice = [0 for _ in range(self.num_dice)]  # Stored $, G, R, E, E, D
        self.dice_left = self.num_dice
        self.turn = 0
        self.score = 0

    def reset(self):
        self.dice = [0 for _ in range(self.num_dice)]
        self.dice_left = self.num_dice

    def count(self):
        if self.dice.count(1) == self.num_dice:
            self.turn += self.one_each
        elif self.num_dice in self.dice:
            self.turn += self.all_same
        else:
            for i in range(self.sides):
                num = self.dice[i] // self.groups[i][0]
                self.turn += self.groups[i][1] * num + self.solo[i] * (self.dice[i] - num)
        self.reset()

    def end_turn(self):
        self.score += self.turn
        self.turn = 0




print(greater_in(1, 3))
print(greater_odds([], [1], 3))
print(greater_odds([1], [1], 6))
print(prob_in(1, 6))
print(dg(4, 6))
print(dg2(4, 6))

