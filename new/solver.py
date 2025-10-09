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
    options: set[State]
    odds: float
    ready: bool = False




@dataclass
class State:
    dice: tuple[int]
    score: int
    greater: set[Edge]
    less: set[Edge]
    value: int
    ready: bool = False



@dataclass
class Solver:
    total_dice: int
    num_faces: int
    scoring_dice: dict[tuple[int], int]
    states: set[State]
    start_state: State
    end_states: set[State]


        

current = 500
expected_value = 100
odds = 0.6

reroll = expected_value * odds > current
