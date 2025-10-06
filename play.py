from random import randrange
import numpy as np


faces = ['G', 'R', 'E', 'e', 'D', '$']
num_faces = 6
total_dice = 6
scores = [
    ([1, 3], [50, 500]),
    ([3], [400]),
    ([3], [300]),
    ([3], [300]),
    ([1, 4], [100, 1000]),
    ([3], [600])
]
scores = [
    [(3, 500), (1, 50)],
    [(3, 400)],
    [(3, 300)],
    [(3, 300)],
    [(4, 1000), (1, 100)],
    [(3, 600)]
]
min_scoring = np.array([1, 3, 3, 3, 1, 3])


def roll(num_dice: int) -> np.array:
    return np.random.randint(num_faces, size=num_dice)


def count(dice: np.array) -> np.array:
    return np.bincount(dice, minlength=num_faces)


def has_scoring(dice: np.array) -> bool:
    return (dice >= min_scoring).any() or (dice == 1).all() or total_dice in dice


def total_scoring(dice: np.array) -> tuple[int, np.array]:
    remaining = np.zeros(num_faces, dtype=int)
    score = 0
    if (dice == 1).all():
        score = 1000
    elif total_dice in dice:
        score = 5000
    else:
        for i in range(len(dice)):
            d = dice[i]
            s = scores[i]
            for j in range(len(s)):
                d2 = s[j][0]
                mod = d // d2
                score += mod * s[j][1]
                d = d % d2
            remaining[i] = d
    return score, remaining


def print_dice(dice: np.array):
    dice.sort()
    print([faces[i] for i in dice])


def get_faces(counted: np.array):
    dice_faces = []
    for i in range(len(counted)):
        d = counted[i]
        dice_faces += faces[i] * d
    return dice_faces


def turn(locked: np.array):
    remaining = total_dice - np.sum(locked)
    rolled = roll(remaining)
    print("Rolled:")
    print_dice(rolled)
    _, trial_dice_left = total_scoring(locked + count(rolled))
    if np.sum(trial_dice_left) == remaining:
        print("No scorining dice, try again")
        return 0, locked
    while True:
        keep_str = input("Enter k/d for each dice to keep/discard: ")
        keep = []
        if len(keep_str) != remaining:
            print("Incorrect number of characters.")
            continue
        for i in range(len(keep_str)):
            char = keep_str[i]
            if char == "k":
                keep.append(rolled[i])
            elif char != "d":
                print(f"Invalid character: {char} at position {i}")
                continue
        if keep:
            trial_counted = locked + count(keep)
            trial_score, trial_dice_left = total_scoring(trial_counted)
            if trial_dice_left.any():
                print("Invalid dice selection, try again.")
            else:
                locked = trial_counted
                return trial_score, trial_counted

total = 0
while total < 5000:
    base_score = 0
    locked = np.zeros(num_faces, dtype=int)
    while True:
        new_score, locked = turn(locked)
        if not new_score:
            break
        if np.sum(locked) == total_dice:
            base_score += new_score
            new_score = 0
            locked = np.zeros(num_faces, dtype=int)
        print(f"Dice: {get_faces(locked)}\nScore: {base_score + new_score}")
        reroll = ""
        while reroll not in ["Y", "N"]:
            reroll = input("Reroll? (y/n): ").upper()
        if reroll == "N":
            total += base_score + new_score
            break
    print(f"\nTotal score: {total}")
            
    

