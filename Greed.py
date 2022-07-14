from math import factorial
import numpy as np
import scipy.special as sp
import time


def odds(n, results, s=6):
    results = [results] if not isinstance(results, list) else results
    total = sum(results)
    if total == 0:
        return 0
    num = 1
    left = n
    for x in results:
        num *= factorial(left) / (factorial(x) * factorial(left - x))
        left -= x
    return num * (1 / s) ** total * ((s - 1) / s) ** (n - total)


def greater_odds(n, greater, other=None, s=6):
    other = [] if other is None else [other] if not isinstance(other, list) else other
    greater = [greater] if not isinstance(greater, list) else greater

    if not greater:
        return odds(n, other, s)

    left = n - sum(other) - sum(greater)
    return sum([greater_odds(n, greater[1:], other + [greater[0] + i], s) for i in range(left+1)])


def below(n, upper, other=None):
    other = [] if other is None else other
    if not upper:
        if np.all(other == 1):
            return []
        else:
            return [other]
    else:
        rolls = []
        start = max(n - sum(upper[1:]), 0)
        stop = min(n+1, upper[0])
        print("start: " + str(start) + " stop: " + str(stop))
        for i in range(start, stop):
            rolls += below(n-i, upper[1:], other + [i])
        return rolls


def between(n, lower, upper=None, other=None, s=6):  # lower and upper must be lists
    other = [] if other is None else other
    upper = np.ones(s, dtype=int) * n if upper is None else upper
    if not lower:
        return 0 if np.all(other) == 1 else odds(n, other, s)
    left = min(n - sum(lower) - sum(other) + 1, upper[0] - lower[0])
    return sum([between(n, lower[1:], upper[1:], other + [lower[0] + i], s) for i in range(left)])


def dg(x, n):
    prob = 0
    for i in range(0, n + 1):
        if i >= x:
            # Roll equal to or more than the required number of dice
            prob += odds(n, i)
        elif i > 0:
            # Roll less than the required number of dice
            prob += odds(n, i) * dg(x - i, n - i)
        elif n > x:
            # Roll none of the required dice, but one of the other kind and keep scoring
            prob += odds(n, 1) * dg(x, n - 1)
    return prob


def six():
    return odds(6, 6) * 4 + (greater_odds(6, 3) - odds(6, 6)) * odds(3, 3) * 4 + dg(6, 6) * 2


def outcomes(n, min_dice):
    combos = [np.arange(0, n+1, i) for i in min_dice]
    combos = np.array(np.meshgrid(*combos)).T.reshape(-1, len(min_dice))
    combos = combos[combos.sum(axis=1) <= n, :]
    return combos


def dice_str(dice):
    return ''.join(map(str, dice))


def link(combos):
    states = {}
    for combo in combos:
        states[dice_str(combo)] = State(combo)
    num_states, _ = combos.shape()
    for i in range(num_states):  #This will go over states with 6 scoring dice, even though they dont necessarily have next states
        combo = combos[i]
        state = states[dice_str(combo)]
        other_states = np.vstack((combos[:i, :], combos[i + 1:, :]))
        rolls = other_states[np.all(other_states >= combo, axis=1)]
        num_rolls, _ = rolls.shape
        for j in range(num_rolls):
            roll = rolls[j]
            # calculate probability of getting roll from combo
            other_rolls = np.vstack((rolls[:j, :], rolls[j+1:, :]))
            sub_states = other_rolls[np.all(other_rolls <= roll, axis=1)]
            state.add_roll(Roll(states[dice_str(roll)], sub_states))


class Roll:
    def __init__(self, state, subs):
        self.state = state
        self.subs = subs
        self.prob = 1
        self.action = -1


class State:
    def __init__(self, dice):
        self.dice = dice
        self.score = 0
        self.rolls = []

    def add_roll(self, roll):
        self.rolls.append(roll)

    def __str__(self):
        return str(self.dice)

    def __repr__(self):
        return str(self.dice)


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
        self.min_dice = [self.groups[i][0] if self.solo[i] == 0 else 1 for i in range(self.sides)]

    def reset(self):
        self.dice = [0 for _ in range(self.num_dice)]
        self.dice_left = self.num_dice

    def roll_probs(self):
        print(outcomes(self.num_dice, self.min_dice))

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





#game = Greed()
#game.roll_probs()
