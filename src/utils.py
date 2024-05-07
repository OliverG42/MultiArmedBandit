import cProfile
import pstats
from decimal import getcontext, Decimal

from numpy import mean, std
import numpy as np
from matplotlib import pyplot as plt

from MultiArmedBandit.src.classes.ColourSink import ColourSink


def plot_graph(data, num_trials, title=None, seed=None):
    error_bar_interval = int(num_trials / 10)
    plt.figure(figsize=(10, 6))

    colour_sink = ColourSink(seed)

    colours = colour_sink.get_colour(num_colours=len(data))

    offset = 0
    for all_cumulative_regrets, choosing_agent in data:
        colour = colours.pop(0)

        avg_cumulative_regrets = mean(all_cumulative_regrets, axis=0)
        std_cumulative_regrets = std(all_cumulative_regrets, axis=0)

        # Plot the smooth curve of average cumulative regrets with error bars
        plt.plot(
            range(0, num_trials),
            avg_cumulative_regrets,
            label=f"{choosing_agent.name}",
            linewidth=1.5,
            color=colour,
            alpha=0.8,
        )

        # Plot the error bars
        # The fmt parameter ensures this doesn't plot the line, which would be jagged
        plt.errorbar(
            range(offset, num_trials, error_bar_interval),
            avg_cumulative_regrets[::error_bar_interval],
            yerr=std_cumulative_regrets[::error_bar_interval],
            fmt=" ",
            color=colour,
            alpha=0.8,
            elinewidth=1,
        )

        # Add horizontal lines at the top and bottom of the error bars
        size = num_trials / 100
        for i in range(offset, num_trials, error_bar_interval):
            x = i
            y = avg_cumulative_regrets[i]
            err = std_cumulative_regrets[i]
            plt.plot(
                [x - size, x + size], [y + err, y + err], color=colour, linewidth=1
            )  # Top line
            plt.plot(
                [x - size, x + size], [y - err, y - err], color=colour, linewidth=1
            )  # Bottom line

        offset += 5

    # Set the x-axis and y-axis limits with some padding
    # This prevents the error bars from protruding into <0, and the y-axis not hitting (0, 0)
    plt.xlim(0, plt.xlim()[1] * 1.01)
    plt.ylim((0, plt.ylim()[1] * 1.01))

    plt.xlabel("Trials")
    plt.ylabel("Cumulative Regret")
    if title is not None:
        plt.title(title, wrap=True)
    plt.legend()
    plt.show()


def profile(function, *args):
    profiler = cProfile.Profile()
    profiler.enable()

    outcome = function(*args)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("tottime")
    stats.print_stats(10)
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(10)
    # noinspection PyUnresolvedReferences
    print("Time taken: " + str(round(stats.total_tt, 3)) + "s")

    return outcome


def lazy_integration(function, *args, step_size=0.001):
    x_values = np.arange(0, 1 + step_size, step_size)
    result = np.sum(function(x_values, *args)) * step_size
    return result


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
