from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


scores = {
    (1, 0, 0, 0, 0, 0): 50,
    (3, 0, 0, 0, 0, 0): 500,
    (0, 3, 0, 0, 0, 0): 400,
    (0, 0, 3, 0, 0, 0): 300,
    (0, 0, 0, 3, 0, 0): 300,
    (0, 0, 0, 0, 1, 0): 100,
    (0, 0, 0, 0, 4, 0): 1000,
    (0, 0, 0, 0, 0, 3): 600,
    (6, 0, 0, 0, 0, 0): 5000,
    (0, 6, 0, 0, 0, 0): 5000,
    (0, 0, 6, 0, 0, 0): 5000,
    (0, 0, 0, 6, 0, 0): 5000,
    (0, 0, 0, 0, 6, 0): 5000,
    (0, 0, 0, 0, 0, 6): 5000,
    (1, 1, 1, 1, 1, 1): 1000,
}

total_dice = 6
num_faces = 6
scores = np.array([
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


@dataclass
class Scoring:
    dice: np.ndarray
    score: int
    greater : list[Greater] = field(default_factory=list)

    def __str__(self):
        return f"dice={self.dice}, score={self.score}, greater={str(self.greater)}"


@dataclass
class Greater:
    score: Scoring
    probability: float = 0.0
    in_between: list[Scoring] = field(default_factory=list)

    def __str__(self):
        return f"score={self.score}, probability={self.probability}, in_between={[i.dice for i in self.in_between]}"



mods = total_dice // scores[:,:-1].sum(axis=1)
ranges = [np.arange(n+1) for n in mods]
meshgrid = np.array(np.meshgrid(*ranges)).T.reshape(-1, len(mods))
combinations = np.matmul(meshgrid, scores)
combinations = combinations[combinations[:,:-1].sum(axis=1) <= total_dice, :]
dice = combinations[:,:-1]
scorings = combinations[:,-1]
unique_scores = np.unique(dice, axis=0)

to_keep = []
scores_list = []
for row in unique_scores:
    mask = (dice == row).all(axis=1)
    max_mask = scorings[mask].argmax()
    max_index = np.where(mask)[0][max_mask]
    """
    duplicate_indices = np.all(dice == row, axis=1).nonzero()[0]
    duplicate_scores = scorings[duplicate_indices]
    max_duplicate_index = duplicate_scores.argmax()
    max_index = duplicate_indices[max_duplicate_index]
    """
    to_keep.append(max_index)
    scores_list.append(Scoring(dice[max_index], scorings[max_index]))

dice = dice[to_keep]
scorings = scorings[to_keep]
combinations = combinations[to_keep]
for i in range(dice.shape[0]):
    row = dice[i]
    greater_indices = (dice >= row).all(axis=1).nonzero()[0]
    greater_indices = greater_indices[greater_indices != i]
    greater_dice = dice[greater_indices]

    greater_list = []
    for j in range(greater_indices.size):
        greater_index = greater_indices[j]
        row2 = greater_dice[j]
        in_between_indices = (greater_dice <= row2).all(axis=1).nonzero()[0]
        in_between_indices = in_between_indices[in_between_indices != j]
        in_between_indices = greater_indices[in_between_indices]
        in_between = [scores_list[k] for k in in_between_indices]
        scores_list[i].greater.append(Greater(scores_list[j], in_between=in_between))


def circularity_check(score: Scoring) -> bool:
    greaters = [score.greater]
    circular = False
    while greaters and not circular:
        greater = greaters.pop()
        for g in greater:
            print(g.score.dice)
            print(g.in_between)
            if score in g.in_between:
                circular = True
                break
            greaters.append(g.in_between)






first = scores_list[0]
print(circularity_check(first))


    
