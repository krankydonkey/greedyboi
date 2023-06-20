from Game import Game, Edge, State
import numpy as np

class Subgame(Game):
    def score(self, dice):
        count = 0
        for i in range(self.s):
            num = dice[i] // self.groups[i][0]
            count += self.groups[i][1] * num + self.solo[i] * (dice[i] - num)
        return count


def stateify(dice, score, greater=None):
    dice_np = np.array(dice)
    if greater:
        rolls = np.array(greater)
    else:
        rolls = np.empty((0, 2), dtype=int)
    return State(dice_np, score, rolls)


def edgeify(n, parent, state, prob, sub_states=None):
    sub_states = [] if sub_states is None else sub_states
    sub_states.append(state)
    edge = Edge(n, parent, state, np.array(sub_states))
    edge.prob = prob
    return edge


state_00 = stateify([0, 0], 0)
state_10 = stateify([1, 0], 25)
state_20 = stateify([2, 0], 200, [[2, 0]])
state_11 = stateify([1, 1], 75)
state_01 = stateify([0, 1], 50)
state_02 = stateify([0, 2], 100, [[0, 2]])

edge_20 = edgeify(2, state_00, state_20, 0.25, [state_10])
edge_11 = edgeify(2, state_00, state_11, 0.5, [state_10, state_01])
edge_02 = edgeify(2, state_00, state_02, 0.25, [state_01])
edge_10_20 = edgeify(2, state_10, state_20, 0.5)
edge_10_11 = edgeify(2, state_10, state_11, 0.5)
edge_01_11 = edgeify(2, state_01, state_11, 0.5)
edge_01_02 = edgeify(2, state_01, state_02, 0.5)

state_00.edges = {"20": edge_20, "11": edge_11, "02": edge_02}
state_10.edges = {"20": edge_10_20, "11": edge_10_11}
state_01.edges = {"02": edge_01_02, "11": edge_01_11}

states = {"00": state_00, "10": state_10, "20": state_20, "11": state_11, "01": state_01, "02": state_02}

game = Subgame(2, 0, 0, ((2, 200), (2, 100)), (25, 50), 5000, 2)
game.link()
print(states)
print(game.states)
print(states == game.states)
print(states["01"] == game.states["01"])
print(states["01"])
print(game.states["01"])