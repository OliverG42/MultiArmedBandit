import functools
import random

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


def bellCurve(x, wins, losses):
    return (x ** wins) * ((1 - x) ** losses)


# Add a dynamic cache from functools
@functools.lru_cache(maxsize=None)
def probSuccessRate(x, wins, losses):
    most_probable = wins / (wins + losses) if wins + losses != 0 else 1

    # Normalise the result
    return bellCurve(x, wins, losses) / bellCurve(most_probable, wins, losses)


class ripple(Agent):
    def __init__(self, arm_state, limit_down=0.01):
        super().__init__()
        self._initialize()
        self.num_arms = arm_state.num_arms
        self.limit_down = limit_down
        self.functions = [
            lambda x, i=i: probSuccessRate(
                x, arm_state.successes[i], arm_state.failures[i]
            )
            for i in range(0, self.num_arms)
        ]

    def chooseLever(self, arm_state):
        arm_value = 1
        result = -1

        while True:
            values = [f(arm_value) for f in self.functions]
            if any(v > self.limit_down for v in values):
                valid_options = [
                    i for i, item in enumerate(values) if item > self.limit_down
                ]
                result = random.choice(valid_options)
                break
            arm_value -= 0.01

        self.functions[result] = lambda x, result=result: probSuccessRate(
            x, arm_state.successes[result], arm_state.failures[result]
        )

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
    assert choice == 1

    doGraph(arm_state.successes, arm_state.failures)
