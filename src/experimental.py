import random

from matplotlib import pyplot as plt
from itertools import combinations
from scipy.optimize import brentq, fsolve
from scipy.special import comb
from ArmState import ArmState


def plotProbSuccess(wins, losses, colour="red", identifier=None):
    if identifier is None:
        identifier = colour

    precision = 4
    precision = 10 ** precision

    plt.plot(
        [x / precision for x in range(0, precision)],
        [probSuccesRate(x / precision, wins, losses) for x in range(0, precision)],
        label=identifier,
        linewidth=1.5,
        color=colour,
        alpha=1,
    )


def bell_curve(x, wins, losses):
    return comb(wins + losses, wins) * (x ** wins) * ((1 - x) ** losses)


def probSuccesRate(x, wins, losses):
    highest = wins / (wins + losses) if wins + losses != 0 else 1
    normalising_factor = 1 / bell_curve(highest, wins, losses)
    return bell_curve(x, wins, losses) * normalising_factor


def computeGradient(func1, func2, x):
    h = 1e-6

    gradient_func1 = (func1(x + h) - func1(x)) / h
    gradient_func2 = (func2(x + h) - func2(x)) / h

    return gradient_func1, gradient_func2


def rippleOld(the_arm_state):
    functions = [lambda x, i=i: probSuccesRate(x, the_arm_state.successes[i], the_arm_state.failures[i]) for i in
                 range(0, the_arm_state.num_arms)]

    pairs = list(combinations(functions, 2))

    rightmost_intersection_pair = [None, None]
    rightmost_intersection_x = -1

    for (f, g) in pairs:
        intersection_x = fsolve(lambda x: f(x) - g(x), 0.5)[0]

        print(functions.index(f), functions.index(g), "intersect at:", intersection_x)

        if intersection_x > rightmost_intersection_x:
            rightmost_intersection_x = intersection_x
            rightmost_intersection_pair = [f, g]

    # No intersections found?
    if rightmost_intersection_x == -1:
        return random.randrange(0, the_arm_state.num_arms)

    gradient_func1, gradient_func2 = computeGradient(rightmost_intersection_pair[0], rightmost_intersection_pair[1],
                                                     rightmost_intersection_x)

    print(rightmost_intersection_x)
    print(gradient_func1, gradient_func2)

    if gradient_func1 > gradient_func2:
        return functions.index(rightmost_intersection_pair[0])
    else:
        return functions.index(rightmost_intersection_pair[1])


def ripple(the_arm_state, limitDown=0.05):
    functions = [lambda x, i=i: probSuccesRate(x, the_arm_state.successes[i], the_arm_state.failures[i]) for i in
                 range(0, the_arm_state.num_arms)]

    arm_value = 1
    while True:
        for f in functions:
            if f(arm_value) > limitDown:
                return functions.index(f)
            arm_value -= 0.01


def doGraph(wins, losses):
    plt.figure(figsize=(10, 6))
    plt.xlabel("Chance of Success")
    plt.ylabel("Probability Guess is Correct")

    colours = ["red", "yellow", "green", "blue", "purple", "pink"]

    for (win, loss, colour) in zip(wins, losses, colours):
        plotProbSuccess(win, loss, colour=colour, identifier=colours.index(colour))

    plt.legend()

    plt.show()


if __name__ == "__main__":
    arm_state = ArmState([0.5, 0.6, 0.4])
    arm_state.successes = [5, 3, 4]
    arm_state.failures = [10, 5, 10]
    arm_state.success_rates = [0.5, 0.6, 0.4]

    print(ripple(arm_state))

    doGraph(arm_state.successes, arm_state.failures)
