from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.special import factorial


@dataclass
class Edge:
    start: State
    end: State
    options: list[State]
    odds: float
    ready: bool = False


@dataclass
class State:
    dice: tuple[int]
    score: int
    greater: list[Edge] = field(default_factory=list)
    less: list[Edge] = field(default_factory=list)
    value: int = 0
    ready: bool = False



class Node:
    def __init__(self, roll: np.ndarray, score: int):
        self.dice = tuple(roll.tolist())
        self.score = score
        self.greater: list[Node] = []
        self.lesser: list[Node] = []

    def __str__(self):
        return f"Dice: {self.dice}, Score: {self.score}, Greater:{[node.dice for node in self.greater]}, Lesser: {[node.dice for node in self.lesser]}"
    
    def __repr__(self):
        return f"Dice: {self.dice}, Score: {self.score}, Greater:{[node.dice for node in self.greater]}, Lesser: {[node.dice for node in self.lesser]}"


def get_roll_odds(num_dice: int, num_faces: int, rolls: np.ndarray):
    dice = rolls
    face_odds = np.full(dice.shape[1], 1 / num_faces)
    return factorial(num_dice) * np.prod(np.power(face_odds, dice), axis=1) / np.prod(factorial(dice), axis=1)


def get_greater_rolls(rolls: np.ndarray, roll: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    greater_equal_indices = (rolls >= roll).all(axis=1).nonzero()[0]
    greater_equal_rolls = rolls[greater_equal_indices]
    greater_indices = (greater_equal_rolls > roll).any(axis=1).nonzero()[0]
    return greater_equal_rolls[greater_indices], greater_equal_indices[greater_indices]


def get_lesser_rolls(rolls: np.ndarray, roll: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lesser_equal_indices = (rolls <= roll).all(axis=1).nonzero()[0]
    lesser_equal_rolls = rolls[lesser_equal_indices]
    lesser_indices = (lesser_equal_rolls < roll).any(axis=1).nonzero()[0]
    return lesser_equal_rolls[lesser_indices], lesser_equal_indices[lesser_indices]
        

def combine(rolls: np.ndarray, scores: np.ndarray, total_dice: int):
    mods = total_dice // rolls.sum(axis=1)
    ranges = [np.arange(n+1) for n in mods]
    meshgrid = np.array(np.meshgrid(*ranges)).T.reshape(-1, len(mods))
    combo_rolls = np.matmul(meshgrid, rolls)
    combo_scores = np.matmul(meshgrid, scores)

    combo_mask = combo_rolls.sum(axis=1) <= total_dice
    combo_rolls = combo_rolls[combo_mask]
    combo_scores = combo_scores[combo_mask]

    unique_rolls = np.unique(combo_rolls, axis=0)
    to_keep = []
    for roll in unique_rolls:
        mask = (combo_rolls == roll).all(axis=1)
        max_mask = combo_scores[mask].argmax()
        max_index = np.where(mask)[0][max_mask]
        to_keep.append(max_index)

    combo_rolls = combo_rolls[to_keep]
    combo_scores = combo_scores[to_keep]
    return combo_rolls, combo_scores


def make_states(combo_rolls: np.ndarray, combo_scores: np.ndarray, total_dice: int, num_faces: int):
    num_states = combo_scores.size
    state_indices = np.arange(num_states)

    # Populate State list
    for i in range(num_states):
        states.append(State(tuple(combo_rolls[i]), int(combo_scores[i])))

    # Link Edges
    for i in range(num_states):
        roll = combo_rolls[i]
        num_dice = roll.sum()

        greater_rolls, greater_indices = get_greater_rolls(combo_rolls, roll)
        greater_state_indices = state_indices[greater_indices]
        odds = get_roll_odds(num_dice, num_faces, greater_rolls - roll)

        for j in range(greater_indices.size):
            greater_roll = greater_rolls[j]
            _, middle_indices = get_lesser_rolls(greater_rolls, greater_roll)
            middle_state_indices = greater_state_indices[middle_indices]

            # Find odds of greater_roll being the greatest scoring roll
            greater_odds = odds[j]
            greatest_rolls, greatest_indices = get_greater_rolls(greater_rolls, greater_roll)
            greatest_odds = odds[greatest_indices]
            # Subtract odds of all combos that are one greater
            for k in range(greatest_indices.size):
                greatest_roll = greatest_rolls[k]
                _, lesser_indices = get_lesser_rolls(greatest_rolls, greatest_roll)
                if lesser_indices.size == 0:
                    greater_odds -= greatest_odds[k] ### WRONG! There sill be overlap in the greater_odds. Need to make full tree, and then for each roll calculate odds of end nodes and then work back

            # Make Edge and link
            edge = Edge(states[i], states[j], [states[k] for k in middle_state_indices], greater_odds)
            edge.start.greater.append(edge)
            edge.end.less.append(edge)

    return states
        
        
def make_tree(combo_rolls: np.ndarray, combo_scores: np.ndarray, total_dice: int, num_faces: int):
    # Sort by number of dice:
    num_dice = combo_rolls.sum(axis=1)
    sorted_indices = np.argsort(num_dice)
    combo_rolls = combo_rolls[sorted_indices]
    combo_scores = combo_scores[sorted_indices]

    # Create nodes
    num_nodes = combo_scores.size
    node_indices = np.arange(num_nodes)
    nodes = [Node(combo_rolls[i], combo_scores[i]) for i in node_indices]

    # Link nodes
    for i in node_indices:
        roll = combo_rolls[i]
        greater_rolls, greater_indices = get_greater_rolls(combo_rolls[i+1:,:], roll)
        greater_node_indices = node_indices[i+1:][greater_indices]  
        for j in range(greater_indices.size):
            _, middle_indices = get_lesser_rolls(greater_rolls[:j-1, :], greater_rolls[j])
            if middle_indices.size == 0:
                lesser_node = nodes[i]
                greater_node = nodes[greater_node_indices[j]]
                lesser_node.greater.append(greater_node)
                greater_node.lesser.append(lesser_node)

    return nodes


def test_dice():
    rolls = np.array([
        [1, 0, 0],
        [1, 1, 1],
        [0, 2, 0],
    ])
    scores = np.array([50, 500, 400])
    total_dice = 3
    num_faces = 3
    return rolls, scores, total_dice, num_faces


def greed_dice():
    rolls = np.array([
        [1, 0, 0, 0, 0, 0],
        [3, 0, 0, 0, 0, 0],
        [0, 3, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0],
        [0, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 4, 0],
        [0, 0, 0, 0, 0, 3],
        [6, 0, 0, 0, 0, 0],
        [0, 6, 0, 0, 0, 0],
        [0, 0, 6, 0, 0, 0],
        [0, 0, 0, 6, 0, 0],
        [0, 0, 0, 0, 6, 0],
        [0, 0, 0, 0, 0, 6],
        [1, 1, 1, 1, 1, 1]
    ])
    scores =  np.array([50, 500, 400, 300, 300, 100, 1000, 600, 5000, 5000, 5000, 5000, 5000, 5000, 1000])
    total_dice = 6
    num_faces = 6
    return rolls, scores, total_dice, num_faces


if __name__ == "__main__":
    rolls, scores, total_dice, num_faces = test_dice()
    #rolls, scores, total_dice, num_faces = greed_dice()
    combo_rolls, combo_scores = combine(rolls, scores, total_dice)
    #states = make_states(combo_rolls, combo_scores, total_dice, num_faces)
    nodes = make_tree(combo_rolls, combo_scores, total_dice, num_faces)
    for n in nodes:
        print(n)