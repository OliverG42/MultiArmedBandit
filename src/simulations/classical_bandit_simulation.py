import multiprocessing

from numpy import cumsum

import Agents
from ArmState import ArmState
from MultiArmedBandit.src import utils

global graph_title


def run_through(agent, arm_state, time_horizon):
    for _ in range(time_horizon):
        chosen_arm = agent.choose_lever(arm_state)
        arm_state.pull_arm(chosen_arm)

    saved_regrets = arm_state.regrets

    # Reset the agent and initial_arm_state
    arm_state.reset()
    agent.reset()

    return saved_regrets


def run_trials(agent, arm_state, time_horizon, num_trials):
    results = []

    for _ in range(num_trials):
        trial_results = run_through(agent, arm_state, time_horizon)

        cumulative_results = cumsum(trial_results)

        results.append(cumulative_results)

    return results


def run_analysis_without_multiprocessing(
    arm_state, agents_list, num_trials, num_samples
):
    results = []

    for agent in agents_list:
        result = run_trials(agent, arm_state, num_trials, num_samples)
        results.append((result, agent))
        print("Finished with agent", agent.name)

    return results


def run_samples_helper(args):
    return run_trials(*args)


def run_analysis_with_multiprocessing(arm_state, agents_list, time_horizon, num_trials):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    inputs = [(agent, arm_state, time_horizon, num_trials) for agent in agents_list]

    results = pool.map(run_samples_helper, inputs)

    pool.close()
    pool.join()

    return list(zip(results, agents_list))


def perform_test(probabilities_list, agents_list, time_horizon, num_trials):
    print(probabilities_list)

    arm_state = ArmState(probabilities_list)

    do_profile: bool = False
    multiprocessing_mode = 0

    analyse_modes = [
        run_analysis_without_multiprocessing,
        run_analysis_with_multiprocessing,
    ]

    if do_profile:
        data = utils.profile(
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

    utils.plot_graph(data, time_horizon, title=graph_title, seed=42)


def get_example(ex_type):
    global graph_title
    if ex_type == 0:
        # Substance Synthesis example
        graph_title = "Substance Synthesis"
        arm_probabilities = [0.1, 0.15, 0.2, 0.25, 0.85, 0.9]
        time_horizon = 1000
        num_trials = 100
    elif ex_type == 1:
        # Consumer Pricing example
        graph_title = "Consumer Pricing"
        arm_probabilities = [
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
            0.02,
            0.03,
            0.04,
            0.05,
            0.09,
        ]
        time_horizon = 100000
        num_trials = 20
    elif ex_type == 2:
        # Freestyle example
        graph_title = "Comparing UCB, BernTS and Ripple against the Substance Synthesis example with double the number of worse arms"
        arm_probabilities = [0.1, 0.15, 0.2, 0.25, 0.1, 0.15, 0.2, 0.25, 0.85, 0.9]
        time_horizon = 2000
        num_trials = 100
    else:
        exit(f"Unknown example type {example_type}")

    return arm_probabilities, time_horizon, num_trials


if __name__ == "__main__":
    example_type = 2

    probabilities, time_horizon, num_trials = get_example(example_type)

    ripple_arm_state = ArmState(probabilities)

    agents = [
        # Agents.CompletelyRandom(),
        # Agents.EpsilonGreedy(epsilon=0.9999, name="Epsilon Greedy with a Geometric Function"),
        # Agents.EpsilonGreedy(epsilon=0.98, name="Linear Function", epsilon_function=epsilon_linear),
        Agents.Ucb(),
        Agents.BernTS(),
        Agents.Ripple(ripple_arm_state),
    ]
    perform_test(probabilities, agents, time_horizon, num_trials)
