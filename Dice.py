from scipy.special import factorial
import numpy as np


def odds(n, results, s=6):
    if results.shape == 0:
        return 0
    left = np.roll(n - np.cumsum(results, axis=1), 1, axis=1)
    left[:, 0] = n
    nums = np.prod(factorial(left) / (factorial(results) * factorial(left - results)), axis=1)
    return sum(nums) * (1 / s) ** n


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


def greater_rolls(n, state, upper, other=None):
    other = [] if other is None else other
    if not upper:
        if all(i == 1 for i in other):
            return []
        else:
            return [other]
    else:
        rolls = []
        start = max(n - sum(state) - sum(upper[1:]), 0) + state[0]
        stop = min(n - sum(state) + 1, upper[0]) + state[0]
        for i in range(start, stop):
            rolls += greater_rolls(n - i, state[1:], upper[1:], other + [i])
        return rolls


def scoring_states(n, min_dice):
    combos = [np.arange(0, n+1, i) for i in min_dice]
    combos = np.array(np.meshgrid(*combos)).T.reshape(-1, len(min_dice))
    combos = combos[combos.sum(axis=1) <= n, :]
    return combos


def dice_str(dice):
    return ''.join(map(str, dice))


def link(n, min_dice):
    combos = scoring_states(n, min_dice)
    states = {}
    for combo in combos:
        states[dice_str(combo)] = State(n, combo, min_dice)
    num_states, _ = combos.shape
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
    return states


class Roll:
    def __init__(self, state, subs):
        self.state = state
        self.subs = subs
        self.prob = 1
        self.action = -1


class State:
    def __init__(self, n, dice, min_dice):
        self.dice = dice
        self.score = 0
        self.rolls = []
        greater = greater_rolls(n, dice, min_dice)
        self.greater = np.array(greater) if greater else np.empty((0, 6), dtype=int)

    def probability(self, n, state):
        return odds(n - sum(state.dice), self.greater - state.dice)

    def add_roll(self, roll):
        self.rolls.append(roll)

    def __str__(self):
        return str(self.dice) + str(self.greater)

    def __repr__(self):
        return str(self.dice) + str(self.greater)


#print(link(5, [1, 1, 3]))
#print(below(2, [1, 3, 3]))
#print(greater_rolls(3, [1, 0, 0], [1, 3, 3]))

#print(odds(3, np.array([[1, 1, 1], [2, 0, 1]])))
