from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.special import factorial


def get_roll_odds(num_dice: int, num_faces: int, rolls: np.ndarray):
    face_odds = np.full(rolls.shape[1], 1 / num_faces)
    return factorial(num_dice) * np.prod(np.power(face_odds, rolls), axis=1) / np.prod(factorial(rolls), axis=1)


def get_greater_rows(matrix: np.ndarray, row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    greater_equal_indices = (matrix >= row).all(axis=1).nonzero()[0]
    greater_equal_rows = matrix[greater_equal_indices]
    greater_indices = (greater_equal_rows > row).any(axis=1).nonzero()[0]
    return greater_equal_rows[greater_indices], greater_indices


def get_lesser_rows(matrix: np.ndarray, row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lesser_equal_indices = (matrix <= row).all(axis=1).nonzero()[0]
    lesser_equal_rows = matrix[lesser_equal_indices]
    lesser_indices = (lesser_equal_rows < row).any(axis=1).nonzero()[0]
    return lesser_equal_rows[lesser_indices], lesser_indices


@dataclass
class SolvedState:
    dice: tuple[int]
    score: int
    reroll: bool
    decisions : dict[tuple[int], SolvedState]

    def best_move(self, roll: tuple[int]) -> SolvedState | None:
        return self.decisions.get(roll)
    
    def expected_value(self) -> float:
        return 0.0

    
def play(zero: SolvedState):
    score = 0
    while True:
        state = zero
        print(f"Total score: {score}")
        while True:
            print(f"Current state: {state.dice}, score: {state.score}, reroll: {state.reroll}")
            dice = input("Enter rolled dice: ") # do parsing to get in tuple form
            if not dice:
                score += state.score
                break
            elif dice in state.decisions:
                state = dice
            else:
                print("Ruh roh, no scoring dice found.")
                break


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


@dataclass
class Solver:
    total_dice: int
    num_faces: int
    base_scores: np.ndarray
    combo_scores: np.ndarray = None
    states: list[State] = field(default_factory=list)
    start_state: State | None = None
    end_states: list[State] = field(default_factory=list)


    def combine(self):
        if self.combo_scores:
            return
        mods = self.total_dice // self.base_scores[:,:-1].sum(axis=1)
        ranges = [np.arange(n+1) for n in mods]
        meshgrid = np.array(np.meshgrid(*ranges)).T.reshape(-1, len(mods))
        combo_scores = np.matmul(meshgrid, self.base_scores)
        combo_scores = combo_scores[combo_scores[:,:-1].sum(axis=1) <= total_dice, :]
        
        dice = combo_scores[:,:-1]
        scores = combo_scores[:,-1]
        unique_scores = np.unique(dice, axis=0)
        to_keep = []
        for row in unique_scores:
            mask = (dice == row).all(axis=1)
            max_mask = scores[mask].argmax()
            max_index = np.where(mask)[0][max_mask]
            to_keep.append(max_index)
        self.combo_scores = combo_scores[to_keep]


    def populate(self):
        for combo in self.combo_scores:
            state = State(tuple(combo[:-1]), int(combo[-1]))
            self.states.append(state)
    

    def link(self):
        rolls = self.combo_scores[:,:-1]
        scores = self.combo_scores[:,-1]
        for i in range(self.combo_scores.shape[0]):
            roll = rolls[i]
            num_dice = np.sum(roll)
            greater_rolls, greater_indices = get_greater_rows(rolls, roll)
            odds = get_roll_odds(num_dice, self.num_faces, greater_rolls - roll)
            for j in range(greater_indices.size):
                # Find options
                greater_index = greater_indices[j]
                greater_roll = greater_rolls[j]
                _, option_indices = get_lesser_rows(greater_rolls, greater_roll)
                option_score_indices = greater_indices[option_indices]

                # Find immediately greater dice
                greatest_rolls, greatest_indices = get_greater_rows(greater_rolls, greater_roll)

                one_greater_indices = []
                for k in greatest_indices:
                    greatest_roll = self.combo_scores[:,:-1][k]
                    _, lesser_indices = get_lesser_rows(greatest_rolls, greatest_roll)
                    lesser_indices = greatest_indices[lesser_indices]

                    if lesser_indices.size == 0:
                        one_greater_indices.append(k)

                greater_odds = odds[j] - np.sum(odds[one_greater_indices])
                
                edge = Edge(
                    self.states[i],
                    self.states[greater_index],
                    [self.states[k] for k in option_score_indices],
                    greater_odds
                )
                edge.start.greater.append(edge)
                edge.end.less.append(edge)



if __name__ == "__main__":
    base_scores = np.array([
        [1, 0, 0, 0, 0, 0, 50],
        [3, 0, 0, 0, 0, 0, 500],
        [0, 3, 0, 0, 0, 0, 400],
        [0, 0, 3, 0, 0, 0, 300],
        [0, 0, 0, 3, 0, 0, 300],
        [0, 0, 0, 0, 1, 0, 100],
        [0, 0, 0, 0, 4, 0, 1000],
        [0, 0, 0, 0, 0, 3, 600],
        [6, 0, 0, 0, 0, 0, 5000],
        [0, 6, 0, 0, 0, 0, 5000],
        [0, 0, 6, 0, 0, 0, 5000],
        [0, 0, 0, 6, 0, 0, 5000],
        [0, 0, 0, 0, 6, 0, 5000],
        [0, 0, 0, 0, 0, 6, 5000],
        [1, 1, 1, 1, 1, 1, 1000]
    ])
    total_dice = 6
    num_faces = 6
    
    base_scores = np.array([
        [1, 0, 0, 50],
        [1, 1, 1, 500],
        [0, 2, 0, 400],
    ])
    total_dice = 3
    num_faces = 3
    
    solver = Solver(total_dice, num_faces, base_scores)
    solver.combine()
    solver.populate()
    solver.link()
