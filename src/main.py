import cProfile
import pstats

import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool

from Agents import bernTS, epsilonGreedy, ucb, softmax, bernGreedy
from ArmState import ArmState
from ColourSink import ColourSink
from experimental import ripple


def runTrials(choosing_agent, arm_state, num_trials):
    for i in range(num_trials):
        chosen_arm = choosing_agent.chooseLever(arm_state)
        arm_state.pull_arm(chosen_arm)

    saved_regrets = arm_state.regrets

    # Reset the agent and arm_state
    choosing_agent.reset()
    arm_state.reset()

    return saved_regrets


def runSamples(choosing_agent, arm_state, num_trials, num_samples):
    results = []

    for _ in range(num_samples):
        trial_results = runTrials(choosing_agent, arm_state, num_trials)

        cumulative_results = np.cumsum(trial_results)

        results.append(cumulative_results)

    return results


def runAnalysisWithoutMultiprocessing(
        arm_state, functions=None, num_trials=100, num_samples=100
):
    if functions is None:
        raise Exception("I didn't receive functions to analyse!")

    results = []

    for choosing_agent in functions:
        result = runSamples(choosing_agent, arm_state, num_trials, num_samples)
        results.append((result, choosing_agent))

    return results


def runSamplesHelper(args):
    return runSamples(*args)


def runAnalysisWithMultiprocessing(
        arm_state, functions=None, num_trials=100, num_samples=100
):
    if functions is None:
        raise Exception("I didn't receive functions to analyse!")

    pool = Pool()

    inputs = [(choosing_agent, arm_state, num_trials, num_samples) for choosing_agent in functions]
    results = pool.map(runSamplesHelper, inputs)

    pool.close()
    pool.join()

    return list(zip(results, functions))


def plotGraph(data, num_trials):
    errorBarInterval = int(num_trials / 10)
    plt.figure(figsize=(10, 6))

    colour_sink = ColourSink()
    colours = colour_sink.getColour(num_colours=len(agents))

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


if __name__ == "__main__":
    probabilities = [x / 10 for x in range(0, 9)]
    arm_state = ArmState(probabilities)
    num_trials = 1000
    num_samples = 100

    agents = [bernTS(), epsilonGreedy(), ucb(), softmax(), bernGreedy(arm_state.num_arms, burn_time=5)]

    do_profile = True
    do_multiprocessing = True

    analyse_modes = [runAnalysisWithoutMultiprocessing, runAnalysisWithMultiprocessing]

    if do_profile:
        data = profile(
            analyse_modes[do_multiprocessing],
            arm_state,
            agents,
            num_trials,
            num_samples,
        )
    else:
        data = analyse_modes[do_multiprocessing](
            arm_state,
            agents,
            num_trials,
            num_samples,
        )

    plotGraph(data, num_trials)
