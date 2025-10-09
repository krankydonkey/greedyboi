from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class SolvedState:
    dice: tuple[int]
    score: int
    reroll: bool
    decisions : dict[tuple[int], SolvedState]

    def best_move(self, roll: tuple[int]) -> SolvedState | None:
        return self.decisions.get(roll)
    
def play(zero: SolvedState):
    score = 0
    while True:
        state = zero
        while True:
            print(f"Current state: {state.dice}, score: {state.score}, reroll: {state.reroll}")
            dice = input("Enter rolled dice: ") # do parsing to get in tuple form
            if not dice:
                score += state.score
                break
            elif dice in state.decisions:
                state = dice
            else:
                break
        
