from numpy import cumsum

from MultiArmedBandit.src.classes.Agents import Uniform, TrackAndStop
from MultiArmedBandit.src.classes.ArmState import ArmState
from MultiArmedBandit.src.utils import plot_graph


def run_through(agent, arm_state, time_horizon):
    toggle_stop = False
    for i in range(time_horizon):
        if toggle_stop:
            chosen_arm = agent.do_pass(arm_state)
        else:
            chosen_arm = agent.choose_lever(arm_state)

            if agent.do_stop(arm_state):
                print("Stopping...")
                print(f"Best arm = {agent.get_result(arm_state)}")
                toggle_stop = True

        arm_state.pull_arm(chosen_arm)

    saved_regrets = arm_state.regrets

    # Reset the agent and initial_arm_state
    arm_state.reset()
    agent.reset()

    return saved_regrets


def run_trials(agent, arm_state, time_horizon, num_trials):
    results = []

    for i in range(num_trials):
        print(f"Trial: {i}")
        trial_results = run_through(agent, arm_state, time_horizon)

        cumulative_results = cumsum(trial_results)

        results.append(cumulative_results)

    return results


def run_analysis(
    arm_state, agents_list, num_trials, num_samples
):
    results = []

    for agent in agents_list:
        result = run_trials(agent, arm_state, num_trials, num_samples)
        results.append((result, agent))
        print("Finished with agent", agent.name)

    return results


def perform_test(probabilities_list, agents_list, time_horizon, num_trials):
    print(probabilities_list)

    arm_state = ArmState(probabilities_list)

    data = run_analysis(
        arm_state,
        agents_list,
        time_horizon,
        num_trials,
    )

    plot_graph(data, time_horizon)


if __name__ == "__main__":
    testing_probabilities = [
        # "Classic" example
        [0.1, 0.15, 0.2, 0.25, 0.9],
        # Very low probabilities
        [0, 0.02, 0.04, 0.06, 0.08],
        # Very high probabilities
        [0.9, 0.92, 0.94, 0.96, 0.98],
        # Middling probabilities
        [0.3, 0.4, 0.5, 0.6, 0.7],
        # A complete mess of probabilities
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        # European and American roulette chances of winning for red/black
        [0.4865, 0.4737],
        # Getting struck by lightning VS Being dealt a Royal Flush VS Bowling a 300-point game
        # All algorithms on average perform terribly, which isn't surprising at all
        [1 / 15300, 1 / 649739, 1 / 11500],
    ]

    time_horizon = 200
    num_trials = 10

    for probabilities in testing_probabilities:
        agents = [
            Uniform(),
            TrackAndStop(failure_probability=0.01),
        ]
        perform_test(probabilities, agents, time_horizon, num_trials)
