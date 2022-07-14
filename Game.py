from scipy.special import factorial
import numpy as np


def roll_odds(n, results, s=6):
    if results.shape == 0:
        return 0
    left = np.roll(n - np.cumsum(results, axis=1), 1, axis=1)
    left[:, 0] = n
    nums = np.prod(factorial(left) / (factorial(results) * factorial(left - results)), axis=1)
    return sum(nums) * (1 / s) ** n


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


def dice_str(dice):
    return ''.join(map(str, dice))


def compare(scores, i, mode):
    if mode not in ['>', '<']:
        _, cols = scores.shape
        return np.empty((0, cols), dtype=int)
    score = scores[i]
    other_scores = np.vstack((scores[:i, :], scores[i + 1:, :]))
    comparison = other_scores >= score if mode is '>' else other_scores <= score
    return other_scores[np.all(comparison, axis=1)]


def greater_in(scores, i):
    other_scores = np.delete(scores, i, 0)
    return other_scores[np.all(other_scores >= scores[i], axis=1)]


class Game:
    def __init__(self, n, min_dice, s=6):
        self.n = n
        self.s = s
        self.min_dice = min_dice
        self.states = {}

    def state(self, score):
        return self.states[dice_str(score)]

    def odds(self, state):
        if state.shape == 0:
            return 0
        left = np.roll(self.n - np.cumsum(state, axis=1), 1, axis=1)
        left[:, 0] = self.n
        nums = np.prod(factorial(left) / (factorial(state) * factorial(left - state)), axis=1)
        return sum(nums) * (1 / self.s) ** self.n

    def greater(self, state):
        rolls = greater_rolls(self.n, state, self.min_dice)
        return np.array(rolls) if rolls else np.empty((0, self.s), dtype=int)

    def scoring(self):
        self.states = {}
        scores = [np.arange(0, self.n + 1, i) for i in self.min_dice]
        scores = np.array(np.meshgrid(*scores)).T.reshape(-1, len(self.min_dice))
        scores = scores[scores.sum(axis=1) <= self.n, :]
        for score in scores:
            rolls = self.greater(score)
            self.states[dice_str(score)] = State(score, rolls)
        return scores

    def link(self):
        scores = self.scoring()
        num_scores, _ = scores.shape
        for i in range(num_scores):  # This will go over states with 6 scoring dice, even though they dont necessarily have next states
            score = scores[i]
            state = self.state(score)
            other_scores = np.delete(scores, i, 0)
            future_scores = other_scores[np.all(other_scores >= score, axis=1)]
            num_futures, _ = future_scores.shape
            for j in range(num_futures):
                future_score = future_scores[j]
                # calculate probability of getting roll from combo
                other_futures = np.delete(future_scores, j, 0)
                sub_states = [self.state(i) for i in other_futures[np.all(other_futures <= future_score, axis=1)]]

                state.add_roll(Edge(self.state(future_score), sub_states))


class Edge:
    def __init__(self, state, sub_states):
        self.state = state
        self.sub_states = sub_states
        self.prob = 1
        self.action = []  # a list of (points, action) pairs for all point totals this state could occur on


class State:
    def __init__(self, dice, rolls):
        self.dice = dice
        self.score = {}  # a dict that maps accumulated points before state to future score weights
        self.rolls = []
        self.rolls = rolls

    def probability(self, n, state):
        return odds(n - sum(state.dice), self.greater - state.dice)

    def add_roll(self, roll):
        self.rolls.append(roll)

    def __str__(self):
        return str(self.dice) + str(self.greater)

    def __repr__(self):
        return str(self.dice) + str(self.greater)


print(link(5, [1, 1, 3]))
# print(below(2, [1, 3, 3]))
# print(greater_rolls(3, [1, 0, 0], [1, 3, 3]))

# print(odds(3, np.array([[1, 1, 1], [2, 0, 1]])))
