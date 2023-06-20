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
    if upper.size == 0:
        if all(i == 1 for i in other):
            return []
        else:
            return [other]
    else:
        rolls = []
        start = max(n - sum(state) - sum(upper[1:]), 0) + state[0]
        stop = min(n - sum(state) + 1, upper[0]) + state[0]
        for i in range(start, stop):
            print(other + [i])
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
    comparison = other_scores >= score if mode == '>' else other_scores <= score
    return other_scores[np.all(comparison, axis=1)]


def greater_in(scores, i):
    other_scores = np.delete(scores, i, 0)
    return other_scores[np.all(other_scores >= scores[i], axis=1)]


class Game:
    def __init__(self, n, all_same, one_each, groups, solo, goal, s=6):
        self.n = n
        self.s = s
        self.states = {}
        self.all_same = all_same
        self.one_each = one_each
        self.groups = groups
        self.solo = solo
        self.goal = goal
        self.min_dice = np.array([self.groups[i][0] if self.solo[i] == 0 else 1 for i in range(self.s)])
        self.scores = self.scoring()
        self.start = self.state(self.scores[-1])

    def state(self, score):
        return self.states[dice_str(score)]

    def score(self, dice):
        count = 0
        if np.count_nonzero(dice == 1) == self.n:
            count += self.one_each
        elif self.n in dice:
            count += self.all_same
        else:
            for i in range(self.s):
                num = dice[i] // self.groups[i][0]
                count += self.groups[i][1] * num + self.solo[i] * (dice[i] - num)
        return count

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
        scores = [np.arange(0, self.n + 1, i) for i in self.min_dice]
        scores = np.array(np.meshgrid(*scores)).T.reshape(-1, len(self.min_dice))
        scores = scores[scores.sum(axis=1) <= self.n, :]
        scores = scores[np.argsort(scores.sum(axis=1))[::-1], :]
        for score in scores:
            rolls = self.greater(score)
            points = self.score(score)
            self.states[dice_str(score)] = State(score, points, rolls)
        return scores

    def max_state(self, dice):
        if np.all(dice == 1):
            return dice
        else:
            score = []
            for i in range(self.s):
                score.append((dice[i] % self.groups[i]) % self.solo[i])
            return score

    def link(self):
        num_scores, _ = self.scores.shape
        for i in range(num_scores):  # This will go over states with 6 scoring dice, even though they dont necessarily have next states
            score = self.scores[i]
            state = self.state(score)
            other_scores = np.delete(self.scores, i, 0)
            future_scores = other_scores[np.all(other_scores >= score, axis=1)]
            num_futures, _ = future_scores.shape
            for j in range(num_futures):
                future_score = future_scores[j]
                other_futures = np.delete(future_scores, j, 0)
                sub_states = [self.state(i) for i in other_futures[np.all(other_futures <= future_score, axis=1)]]
                edge = Edge(self.n, state, self.state(future_score), sub_states)
                state.add_edge(edge)

    def max_combos(self):
        return self.scores[np.sum(self.scores, axis=1) == self.n]

    def propagate(self, score):
        for dice, state in self.states.items():
            state.add_score(score)

    def play(self):
        score = 0
        state = self.start
        command = input("Dice rolled ($GREED): ")
        while command != "quit":
            if command == "":
                self.play()
                return
            else:
                dice = np.copy(state.dice)
                dice += np.array([int(i) for i in command])
                max_state = dice // self.min_dice
                print(state)
                edge = state.edges[dice_str(max_state)]
                next_state = edge.action[score]  # Calculate scoring state
                print("Keep: " + str(next_state.dice - state.dice))
                print("Roll" if next_state.roll(score) else "Stop")
                state = next_state
                command = input("Dice rolled: ")


class Edge:
    def __init__(self, n, parent, state, sub_states):
        self.prob = state.probability(n, parent)
        self.state = state
        self.sub_states = sub_states
        self.action = {}  # a dict of points:state pairs for all point totals this state could occur on

    def add_action(self, score):
        best = 0
        best_state = None
        for state in self.sub_states:
            weighted = state.scores[score]
            if weighted > best:
                best = weighted
                best_state = state
        self.action[score] = best_state
        return self.prob * best

    def __str__(self):
        return str(self.state.dice)

    def __eq__(self, other):
        return isinstance(other, Edge) and self.prob == other.prob and self.state == other.state \
               and np.array_equal(self.sub_states, other.sub_states) and self.action == other.action


class State:
    def __init__(self, dice, score, rolls):
        self.dice = dice
        self.score = score
        self.scores = {}  # a dict that maps accumulated points before state to future score weights
        self.edges = {}
        self.rolls = rolls

    def probability(self, n, parent):
        return roll_odds(n - sum(parent.dice), self.rolls - parent.dice)

    def add_edge(self, edge):
        self.edges[dice_str(edge.state.dice)] = edge

    def add_score(self, score):
        weighted = 0
        for key, edge in self.edges.items():
            weighted += edge.add_action(score)
        normal = score + self.score
        self.scores[score] = weighted if weighted > normal else normal

    def roll(self, score):
        return self.scores[score] > score + self.score

    def __str__(self):
        return str(self.dice) + str(self.edges.keys())

    def __repr__(self):
        return str(self.dice) + str(self.rolls)

    def __eq__(self, other):
        return isinstance(other, State) and np.array_equal(self.dice, other.dice) and self.score == other.score \
               and self.scores == other.scores and self.edges == other.edges and self.rolls == other.rolls


#greed = Game(6, 5000, 1000, ((3, 600), (3, 500), (3, 400), (3, 300), (3, 300), (4, 1000)), (0, 50, 0, 0, 0, 100), 5000)
#greed.link()
#greed.propagate(0)
#greed.play()
print(greater_rolls(5, np.array([2, 1, 0, 0, 0, 0]), np.array([1, 1, 3, 3, 3, 3])))