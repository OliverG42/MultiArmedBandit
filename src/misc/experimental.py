import functools
from decimal import Decimal, getcontext

import numpy as np
from matplotlib import pyplot as plt

from Agents import Agent
from ArmState import ArmState
from utils import lazy_integration


def plot_prob_success(wins, losses, colour, identifier=None):
    if identifier is None:
        identifier = colour

    precision = 4
    precision = 10**precision

    plt.plot(
        [x / precision for x in range(0, precision)],
        [prob_success_rate(x / precision, wins, losses) for x in range(0, precision)],
        label=identifier,
        linewidth=1.5,
        color=colour,
        alpha=1,
    )


# Essentially doing (x ** wins) * ((1 - x) ** losses), but avoiding errors and very small number rounding
def bell_curve(x, wins, losses):
    # Set the precision of Decimal
    getcontext().prec = 5

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
        first = decimal_x**decimal_wins

    # Catch cases when calculating 0^0=1 or 0^n=0, which causes Decimal to freak out
    if (1 - decimal_x) == 0:
        if decimal_losses == 0:
            second = 1
        else:
            second = 0
    else:
        second = (1 - decimal_x) ** decimal_losses

    return float(first * second)


def bell_curve_vectorized(x, wins, losses):
    first = np.where(x == 0, np.where(wins == 0, 1, 0), x**wins)
    second = np.where((1 - x) == 0, np.where(losses == 0, 1, 0), (1 - x) ** losses)

    return first * second


def prob_success_rate(x, wins, losses):
    most_probable = wins / (wins + losses) if wins + losses != 0 else 1

    # Normalise the result
    denominator = bell_curve(most_probable, wins, losses)
    if denominator == 0:
        return 0
    return bell_curve(x, wins, losses) / bell_curve(most_probable, wins, losses)


# Stands for Really Accurate Gambler
class Rag(Agent):
    def __init__(self):
        super().__init__()
        self._initialize()

    # Add a dynamic cache from functools
    @functools.lru_cache(maxsize=None)
    def expectedRewards(self, successes, failures):
        normalisation_factor = 1 / lazy_integration(
            bell_curve_vectorized, successes, failures
        )
        x_values = np.arange(0, 1.001, 0.001)
        integrand = (
            x_values
            * bell_curve_vectorized(x_values, successes, failures)
            * normalisation_factor
        )
        result = np.trapz(integrand, dx=0.001)
        return result

    def choose_lever(self, the_arm_state):
        expectedNormalisedRewards = np.array(
            [
                self.expectedRewards(s, f)
                for (s, f) in zip(the_arm_state.successes, the_arm_state.failures)
            ]
        )
        result = np.argmax(expectedNormalisedRewards)

        return result


class Cliff(Agent):
    def __init__(self, takeover):
        super().__init__()
        self._initialize()
        self.takeover = takeover

    def choose_lever(self, the_arm_state):
        greedy_arm = np.argmax(the_arm_state.success_rates)
        greedy_success_rate = the_arm_state.success_rates[greedy_arm]
        result = greedy_arm

        for arm_index, (successes, failures) in enumerate(
            zip(the_arm_state.successes, the_arm_state.failures)
        ):
            if arm_index == greedy_arm:
                continue

            if (
                prob_success_rate(greedy_success_rate, successes, failures)
                > self.takeover
            ):
                result = arm_index

        return result


def do_graph(wins, losses):
    plt.figure(figsize=(10, 6))
    plt.xlabel("Chance of Success - p")
    plt.subplots_adjust(left=0.2)
    plt.ylabel("\u2119(wins,losses | p)", rotation=0, ha="right")

    colours = ["red", "green", "blue"]

    for win, loss, colour in zip(wins, losses, colours):
        plot_prob_success(win, loss, colour=colour, identifier=colours.index(colour))

    plt.axhline(y=0.05, color="black", linestyle="--", label="Tolerance")

    plt.legend()

    plt.show()


if __name__ == "__main__":
    arm_state = ArmState([0.5, 0.6, 0.4])
    arm_state.successes = [8, 5, 4]
    arm_state.failures = [13, 9, 10]
    arm_state.success_rates = np.array(arm_state.successes) / np.array(
        arm_state.failures
    )

    rippleAgent = Ripple(arm_state)
    choice = rippleAgent.choose_lever(arm_state)
    print(choice)
    assert choice == 1

    do_graph(arm_state.successes, arm_state.failures)
