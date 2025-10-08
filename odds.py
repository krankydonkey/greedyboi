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
    pass


def at_least2(dice: np.ndarray, num_dice: int, num_faces: int) -> float:
    pass