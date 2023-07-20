import functools
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt

from Agents import Agent
from ArmState import ArmState


def plotProbSuccess(wins, losses, colour="black", identifier=None):
    if identifier is None:
        identifier = colour

    precision = 4
    precision = 10**precision

    plt.plot(
        [x / precision for x in range(0, precision)],
        [probSuccessRate(x / precision, wins, losses) for x in range(0, precision)],
        label=identifier,
        linewidth=1.5,
        color=colour,
        alpha=1,
    )


def bellCurve(x, wins, losses):
    return (x**wins) * ((1 - x) ** losses)


def probSuccessRate(x, wins, losses):
    most_probable = wins / (wins + losses) if wins + losses != 0 else 1

    # Normalise the result
    return bellCurve(x, wins, losses) / bellCurve(most_probable, wins, losses)


class ripple(Agent):
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
        arm_value = 1

        while arm_value >= 0:
            success_rate = probSuccessRate(arm_value, successes, failures)

            if success_rate > self.limit_down:
                return arm_value

            arm_value -= self.search_increment

        return None

    def _doesNotWorkFindIntersection(self, successes, failures):
        left = 0
        right = 1

        while left <= right:
            mid = (left + right) / 2
            success_rate = probSuccessRate(mid, successes, failures)

            if success_rate > self.limit_down:
                right = mid - self.search_increment
            else:
                left = mid + self.search_increment

        return right

    def chooseLever(self, arm_state):
        # Find which intersection point(s) has changed
        for i in range(0, self.num_arms):
            if self.arm_pulls_memory[i] != arm_state.arm_pulls[i]:
                # Change the intersection point
                self.intersection_points[i] = self._findIntersection(
                    arm_state.successes[i], arm_state.failures[i]
                )

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

    rippleAgent = ripple(arm_state)
    choice = rippleAgent.chooseLever(arm_state)
    print(choice)
    assert choice == 1

    doGraph(arm_state.successes, arm_state.failures)
