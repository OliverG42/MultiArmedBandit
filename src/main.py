import cProfile
import multiprocessing
import pstats

import matplotlib.pyplot as plt
import numpy as np

import concurrent.futures

from algorithms import bernTS, bernGreedy, epsilonGreedy, ucb, softmax
from experimental import ripple
from ArmState import ArmState


def perform_sampling(choosing_function, reward_probs, num_trials):
    arm_state = ArmState(reward_probs)

    for _ in range(num_trials):
        chosen_arm = choosing_function(arm_state)
        arm_state.pull_arm(chosen_arm)

    return arm_state.regrets


def compute(choosing_function, reward_probs, num_trials, num_samples):
    return [
        np.cumsum(perform_sampling(choosing_function, reward_probs, num_trials))
        for _ in range(num_samples)
    ]


def plot(
    choosing_function,
    colour,
    num_trials,
    errorBarInterval,
    offset,
    all_cumulative_regrets,
):
    avg_cumulative_regrets = np.mean(all_cumulative_regrets, axis=0)
    std_cumulative_regrets = np.std(all_cumulative_regrets, axis=0)

    # Plot the smooth curve of average cumulative regrets with error bars
    plt.plot(
        range(0, num_trials),
        avg_cumulative_regrets,
        label=f"{choosing_function.__name__}",
        linewidth=1.5,
        color=colour,
        alpha=0.8,
    )

    # Plot the error bars
    # The fmt parameter ensures this doesn't plot the line, which would be jagged
    plt.errorbar(
        range(offset, num_trials, errorBarInterval),
        avg_cumulative_regrets[::errorBarInterval],
        yerr=std_cumulative_regrets[::errorBarInterval],
        fmt=" ",
        color=colour,
        alpha=0.8,
    )


def makeGraph(reward_probs):
    print(reward_probs)
    plt.figure(figsize=(10, 6))

    functions = [
        bernTS,
        bernGreedy,
        epsilonGreedy,
        ucb,
        softmax,
        ripple
    ]

    colours = ["red", "yellow", "green", "blue", "purple", "pink"]

    num_trials = 1000  # How many "lever pulls" are there?
    num_samples = 100  # How many "runs" are there?
    errorBarInterval = int(num_trials / 10)

    # Adjust as needed
    workers = multiprocessing.cpu_count()

    # Create a thread pool executor
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []

        for choosing_function, colour in zip(functions, colours):
            future = executor.submit(
                compute, choosing_function, reward_probs, num_trials, num_samples
            )
            futures.append((future, choosing_function, colour))
            plot(
                choosing_function,
                colour,
                num_trials,
                errorBarInterval,
                colours.index(colour),
                future.result(),
            )

        # Wait for all the computations to complete
        concurrent.futures.wait([future for future, _, _ in futures])

    # Set the x-axis and y-axis limits with some padding
    # This prevents the error bars from protruding into <0, and the y-axis not hitting (0, 0)
    plt.xlim(0, plt.xlim()[1] * 1.01)
    plt.ylim((0, plt.ylim()[1] * 1.01))

    plt.xlabel(f"Trials {reward_probs}")
    plt.ylabel("Cumulative Regret")
    plt.legend()


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    makeGraph([0.4, 0.45, 0.5])
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("tottime")
    stats.print_stats(10)
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(10)
    print(stats.total_tt)
    plt.show()
    # plt.savefig("graph.png")
