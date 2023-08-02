import functools
from copy import deepcopy
from decimal import Decimal, getcontext

import numpy as np
from matplotlib import pyplot as plt

from Agents import Agent
from ArmState import ArmState


def plotProbSuccess(wins, losses, colour="black", identifier=None):
    if identifier is None:
        identifier = colour

    precision = 4
    precision = 10 ** precision

    plt.plot(
        [x / precision for x in range(0, precision)],
        [probSuccessRate(x / precision, wins, losses) for x in range(0, precision)],
        label=identifier,
        linewidth=1.5,
        color=colour,
        alpha=1,
    )


def oldBellCurve(x, wins, losses):
    return (x ** wins) * ((1 - x) ** losses)


def bellCurve(x, wins, losses):
    getcontext().prec = 5  # Set the precision as needed

    decimal_x = Decimal(x)
    decimal_wins = Decimal(wins)
    decimal_losses = Decimal(losses)

    # Catch cases when calculating 0^0=1 or 0^n=0, which causes Decimal to freak out
    if decimal_x == 0:
        if decimal_wins == 0:
            first = 1
        else:
            first = 0
    else:
        first = decimal_x ** decimal_wins

    # Catch cases when calculating 0^0=1 or 0^n=0, which causes Decimal to freak out
    if (1 - decimal_x) == 0:
        if decimal_losses == 0:
            second = 1
        else:
            second = 0
    else:
        second = (1 - decimal_x) ** decimal_losses

    probability = first * second

    """correct = oldBellCurve(x, wins, losses)

    if abs(probability - Decimal(correct)) > 5e-3:
        print(f"{probability} is not close to {correct}")
        print(f"{first} compared to ({x} ** {wins}) = {(x ** wins)}")
        print(f"{second} compared to ({1 - x} ** {losses}) = {((1 - x) ** losses)}")
        print(f"With inputs {x}, {wins} and {losses}")
        sys.exit()"""

    return probability


def probSuccessRate(x, wins, losses):
    most_probable = wins / (wins + losses) if wins + losses != 0 else 1

    # Normalise the result
    return float(bellCurve(x, wins, losses) / bellCurve(most_probable, wins, losses))


class Ripple(Agent):
    def __init__(self, arm_state, limit_down=0.01):
        super().__init__()
        self._initialize()
        self.name = self.name + " " + str(limit_down)
        self.num_arms = arm_state.num_arms
        self.limit_down = limit_down
        self.search_increment = 0.001

        self.intersection_points = [
            self._findIntersection(
                deepcopy(arm_state.successes[i]), deepcopy(arm_state.failures[i])
            )
            for i in range(0, self.num_arms)
        ]
        self.arm_pulls_memory = deepcopy(arm_state.arm_pulls)

    # Add a dynamic cache from functools
    @functools.lru_cache(maxsize=None)
    def _findIntersection(self, successes, failures):
        upper = 1
        lower = successes / (successes + failures) if successes + failures != 0 else 1

        accuracy = 1e-2

        while upper - lower > accuracy:
            middle = (upper + lower) / 2
            success_rate = probSuccessRate(middle, successes, failures)

            if success_rate < self.limit_down:
                upper = middle
            else:
                lower = middle

        return lower

    @functools.lru_cache(maxsize=None)
    def _betterFindIntersection(self, successes, failures):
        arm_value = successes / (successes + failures) if successes + failures != 0 else 1

        while arm_value < 1:
            success_rate = probSuccessRate(arm_value, successes, failures)

            if success_rate < self.limit_down:
                return arm_value

            arm_value += self.search_increment

        return 1

    def chooseLever(self, arm_state):
        # Find which intersection point(s) has changed
        for i in range(0, self.num_arms):
            if self.arm_pulls_memory[i] != arm_state.arm_pulls[i]:
                # Change the intersection point
                old = self._findIntersection(
                    arm_state.successes[i], arm_state.failures[i]
                )
                """new = self._betterFindIntersection(
                    arm_state.successes[i], arm_state.failures[i]
                )"""

                """if not np.isclose(old, new, atol=1e-2):
                    print(old, new)
                    exit(123)"""

                self.intersection_points[i] = old

        result = np.argmax(self.intersection_points)

        return result


def doGraph(wins, losses):
    plt.figure(figsize=(10, 6))
    plt.xlabel("Chance of Success - p")
    plt.subplots_adjust(left=0.2)
    plt.ylabel("\u2119(wins,losses | p)", rotation=0, ha="right")

    colours = ["red", "yellow", "green"]

    for win, loss, colour in zip(wins, losses, colours):
        plotProbSuccess(win, loss, colour=colour, identifier=colours.index(colour))

    plt.legend()

    plt.show()


if __name__ == "__main__":
    arm_state = ArmState([0.5, 0.6, 0.4])
    arm_state.successes = [5, 3, 4]
    arm_state.failures = [10, 5, 10]
    arm_state.success_rates = [0.5, 0.6, 0.4]

    rippleAgent = Ripple(arm_state)
    choice = rippleAgent.chooseLever(arm_state)
    print(choice)
    assert choice == 1

    doGraph(arm_state.successes, arm_state.failures)
