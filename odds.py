from scipy.special import factorial
import numpy as np
import timeit

# Calculates different dice odds using multinomial distribution

def multinomial(num_dice: int, combinations: np.ndarray, probabilities: np.ndarray) -> float:
    if combinations.ndim == 1:
        return factorial(num_dice) * np.prod(np.power(probabilities, combinations)) / (np.prod(factorial(combinations)))
    else:
        odds = factorial(num_dice) * np.prod(np.power(probabilities, combinations), axis=1) / np.prod(factorial(combinations), axis=1)
        return np.sum(odds)


def at_least(num_dice: int, dice: np.ndarray, num_faces: int = 6) -> float:
    """
    Calculates the probability of rolling at least the given dice. If num_dice is greater than sum(dice),
    then the probability of rolling more than that given dice will be included.

    :param int num_dice: the number of dice rolled
    :param np.ndarray dice: the rolled dice, as a count of each face. For example,
                            rolling two of one face and one of another would be [2, 1]
    :param int num_sides: the number of sides on each dice
    """
    probabilities = np.full(dice.size, 1 / num_faces)
    return multinomial(num_dice, dice, probabilities)


def exact(num_dice: int, dice: np.ndarray, max_dice: np.ndarray | None = None, num_faces: int = 6) -> float:
    """
    Calculates the probability of rolling exactly the given dice. The remaing rolls would consist of a
    random combination of all unspecified faces. An upper bound for the ocurences of each face can
    optionally be given; in this case, the probability of all valid combinations between the desired
    dice and the upper bound will be calculated and summed to give a total probability.

    :param int num_dice: the number of dice rolled
    :param np.ndarray dice: the rolled dice, as a count of each face. For example,
                            rolling two of one face and one of another would be [2, 1]
    :param np.ndarray max_dice: an upper bound for the rolled dice, as a count of each face.
    :param int num_sides: the number of sides on each dice
    """

    if max_dice is None:
        combinations = np.expand_dims(dice, axis=0)
    else:
        # Calculate all possible combinations within the bounds of max_dice
        diff = max_dice - dice
        ranges = [np.arange(n+1) for n in diff]
        combinations = np.array(np.meshgrid(*ranges)).T.reshape(-1, len(diff))
        combinations += dice

    # If a number of faces have not been specified, create another column in the combination
    # matrix and assign all remaining dice to this group of faces. If no faces have been left
    # unspecified, then only combinations that add up to num_dice are valid.
    if dice.size < num_faces:
        combinations = combinations[combinations.sum(axis=1) <= num_dice, :]
        remaining_dice = num_dice - combinations.sum(axis=1)
        combinations = np.column_stack((combinations, remaining_dice))
        probabilities = np.full(combinations.shape, 1 / num_faces)
        probabilities[:,-1] = (num_faces - dice.size) / num_faces
    else:
        combinations = combinations[combinations.sum(axis=1) == num_dice, :]

    return multinomial(num_dice, combinations, probabilities)
