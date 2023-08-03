import cProfile
import multiprocessing
import pstats
import random

import matplotlib.pyplot as plt
import dask.bag as db
from dask.distributed import Client

from Agents import *
from ArmState import ArmState
from ColourSink import ColourSink
from experimental import Ripple


def run_through(choosing_agent, arm_state, time_horizon):
    for i in range(time_horizon):
        chosen_arm = choosing_agent.chooseLever(arm_state)
        arm_state.pull_arm(chosen_arm)

    saved_regrets = arm_state.regrets

    # Reset the agent and initial_arm_state
    arm_state.reset()
    choosing_agent.reset()

    return saved_regrets


def runTrials(choosing_agent, arm_state, time_horizon, num_trials):
    results = []

    for _ in range(num_trials):
        trial_results = run_through(choosing_agent, arm_state, time_horizon)

        cumulative_results = np.cumsum(trial_results)

        results.append(cumulative_results)

    return results


def runAnalysisWithoutMultiprocessing(
        arm_state, functions, num_trials, num_samples
):
    results = []

    for choosing_agent in functions:
        result = runTrials(choosing_agent, arm_state, num_trials, num_samples)
        results.append((result, choosing_agent))
        print("Finished with agent", choosing_agent.name)

    return results


def runSamplesHelper(args):
    return runTrials(*args)


def runAnalysisWithMultiprocessing(
        arm_state, agents, time_horizon, num_trials
):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    inputs = [
        (choosing_agent, arm_state, time_horizon, num_trials)
        for choosing_agent in agents
    ]

    results = pool.map(runSamplesHelper, inputs)

    pool.close()
    pool.join()

    return list(zip(results, agents))


def plotGraph(data, num_trials):
    errorBarInterval = int(num_trials / 10)
    plt.figure(figsize=(10, 6))

    colour_sink = ColourSink()

    colours = colour_sink.getColour(num_colours=len(data))

    offset = 0
    for all_cumulative_regrets, choosing_agent in data:
        colour = colours.pop(0)

        avg_cumulative_regrets = np.mean(all_cumulative_regrets, axis=0)
        std_cumulative_regrets = np.std(all_cumulative_regrets, axis=0)

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
            range(offset, num_trials, errorBarInterval),
            avg_cumulative_regrets[::errorBarInterval],
            yerr=std_cumulative_regrets[::errorBarInterval],
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


def performTest(probabilities, agents):
    print(probabilities)
    time_horizon = 400
    num_trials = 100

    arm_state = ArmState(probabilities)

    do_profile = True
    multiprocessing_mode = 1

    analyse_modes = [runAnalysisWithoutMultiprocessing, runAnalysisWithMultiprocessing]

    if do_profile:
        data = profile(
            analyse_modes[multiprocessing_mode],
            arm_state,
            agents,
            time_horizon,
            num_trials,
        )
    else:
        data = analyse_modes[multiprocessing_mode](
            arm_state,
            agents,
            time_horizon,
            num_trials,
        )

    plotGraph(data, time_horizon)


if __name__ == "__main__":
    testing_probabilities = [
        # "Classic" example
        [0.1, 0.15, 0.2, 0.25, 0.1, 0.15, 0.2, 0.25, 0.9],
        # Very low probabilities
        [float("{:.3f}".format(random.uniform(0, 0.1))) for _ in range(5)],
        # Very high probabilities
        [float("{:.3f}".format(random.uniform(0.9, 0.999))) for _ in range(5)],
        # Middling probabilities
        [float("{:.3f}".format(random.uniform(0.3, 0.7))) for _ in range(5)],
        # A complete mess of probabilities
        [float("{:.3f}".format(random.uniform(0, 1))) for _ in range(5)],
        # Europe and American roulette red/black
        [0.4865, 0.4737],
        # Overloading with way too many options
        [float("{:.3f}".format(random.uniform(0, 1))) for _ in range(50)],
        # Getting struck by lightning VS Being dealt a Royal Flush VS Bowling a 300-point game
        # All algorithms on average perform terribly, which isn't surprising at all
        [1 / 15300, 1 / 649739, 1 / 11500],
    ]

    for probabilities in testing_probabilities:
        agents = [
            Ripple(ArmState(probabilities), limit_down=0.05),
            Ucb(),
            # BernTS(),
        ]
        performTest(probabilities, agents)
