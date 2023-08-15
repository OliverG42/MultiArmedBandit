import cProfile
import pstats

from numpy import mean, std
import numpy as np
from matplotlib import pyplot as plt

from src.ColourSink import ColourSink


def plot_graph(data, num_trials):
    error_bar_interval = int(num_trials / 10)
    plt.figure(figsize=(10, 6))

    colour_sink = ColourSink()

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
        )

        offset += 1

    # Set the x-axis and y-axis limits with some padding
    # This prevents the error bars from protruding into <0, and the y-axis not hitting (0, 0)
    plt.xlim(0, plt.xlim()[1] * 1.01)
    plt.ylim((0, plt.ylim()[1] * 1.01))

    plt.xlabel("Trials")
    plt.ylabel("Cumulative Regret")
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
