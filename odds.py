from scipy.special import factorial
import numpy as np
import timeit

# Calculates different dice odds using multinomial distribution

def multinomial(num_dice: int, results: np.ndarray, probabilities: np.ndarray) -> float:
    return factorial(num_dice) * np.prod(np.power(probabilities, results)) / (np.prod(factorial(results)))


def at_least(dice: np.ndarray, num_dice: int, num_faces: int) -> float:
    """
    Calculates the probability of rolling at least the given dice. If num_dice is greater than sum(dice),
    then the probability of rolling more than that given dice will be included.

    :param np.ndarray dice: the rolled dice, as a count of each face. For example,
                            rolling two of one face and one of another would be [2, 1]
    :param int num_dice: the number of dice rolled
    """
    probabilities = np.full(dice.size, 1 / num_faces)
    return multinomial(num_dice, dice, probabilities)


def exact(dice: np.ndarray, num_dice: int, num_faces: int) -> float:
    """
    Calculates the probability of rolling exactly the given dice. The remaing rolls would consist of a
    random combination of all unspecified faces.

    :param np.ndarray dice: the rolled dice, as a count of each face. For example,
                            rolling two of one face and one of another would be [2, 1]
    :param int num_dice: the number of dice rolled
    """
    probabilities = np.full(dice.size + 1, 1 / num_faces)
    probabilities[-1] = (num_faces - dice.size) / num_faces
    results = np.append(dice, num_dice - np.sum(dice))
    return multinomial(num_dice, results, probabilities)


def exact_less(dice: np.ndarray, less_than: np.ndarray, num_dice: int, num_faces: int) -> float:
    ranges = [np.arange(n) for n in less_than]
    combinations = np.array(np.meshgrid(*ranges)).T.reshape(-1, len(less_than))
    combinations += dice
    combinations = combinations[combinations.sum(axis=1) <= num_dice, :]
    remaining_dice = num_dice - combinations.sum(axis=1)
    combinations = np.column_stack((combinations, remaining_dice))
    probabilities = np.full(combinations.shape, 1 / num_faces)
    probabilities[:,-1] = (num_faces - dice.size) / num_faces
    totals = factorial(num_dice) * np.prod(np.power(probabilities, combinations), axis=1) / np.prod(factorial(combinations), axis=1)
    return np.sum(totals)
    #scores = scores[scores.sum(axis=1) <= num_dice, :]
    #scores = scores[np.argsort(scores.sum(axis=1))[::-1], :]


def prob(dice: np.ndarray, max_dice: np.ndarray | None = None, num_dice: int | None = None, num_faces: int = 6):
        pass


exact_less(np.array([2, 2]), np.array([2, 3]), 6, 6)