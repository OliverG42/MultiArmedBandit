import multiprocessing
import random
import sys

from Agents import *
from ArmState import ArmState
from experimental import Ripple
from utils import *


def run_through(agent, arm_state, time_horizon):
    for i in range(time_horizon):
        chosen_arm = agent.chooseLever(arm_state)
        arm_state.pull_arm(chosen_arm)

    saved_regrets = arm_state.regrets

    # Reset the agent and initial_arm_state
    arm_state.reset()
    agent.reset()

    return saved_regrets


def runTrials(agent, arm_state, time_horizon, num_trials):
    results = []

    for _ in range(num_trials):
        trial_results = run_through(agent, arm_state, time_horizon)

        cumulative_results = np.cumsum(trial_results)

        results.append(cumulative_results)

    return results


def runAnalysisWithoutMultiprocessing(
        arm_state, agents_list, num_trials, num_samples
):
    results = []

    for agent in agents_list:
        result = runTrials(agent, arm_state, num_trials, num_samples)
        results.append((result, agent))
        print("Finished with agent", agent.name)

    return results


def runSamplesHelper(args):
    return runTrials(*args)


def runAnalysisWithMultiprocessing(
        arm_state, agents_list, time_horizon, num_trials
):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    inputs = [
        (agent, arm_state, time_horizon, num_trials)
        for agent in agents_list
    ]

    results = pool.map(runSamplesHelper, inputs)

    pool.close()
    pool.join()

    return list(zip(results, agents_list))


def performTest(probabilities_list, agents_list):
    print(probabilities_list)
    time_horizon = 400
    num_trials = 200

    arm_state = ArmState(probabilities_list)

    do_profile = True
    multiprocessing_mode = 0

    analyse_modes = [runAnalysisWithoutMultiprocessing, runAnalysisWithMultiprocessing]

    if do_profile:
        data = profile(
            analyse_modes[multiprocessing_mode],
            arm_state,
            agents_list,
            time_horizon,
            num_trials,
        )
    else:
        data = analyse_modes[multiprocessing_mode](
            arm_state,
            agents_list,
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
        sys.exit()
