from random import randint

import numpy as np


# TODO Add "burn-in" time
# TODO Try each arm x times, then switch to greedy
def bernGreedy(arm_state):
    return np.argmax(arm_state.success_rates)


def bernTS(arm_state, alphaPrior=1, betaPrior=1):
    samples = np.random.beta(
        arm_state.successes + alphaPrior, arm_state.failures + betaPrior
    )
    return np.argmax(samples)


def epsilonGreedy(arm_state, epsilon=0.9):
    # Slowly decrease the probability to do something random
    epsilon = epsilon * (0.9**arm_state.total_pulls)

    if np.random.random() < epsilon:
        # Choose at random
        chosen_arm = np.random.randint(arm_state.num_arms)
    else:
        # Choose greedily
        chosen_arm = np.argmax(arm_state.success_rates)

    return chosen_arm


def completelyRandom(arm_state):
    return randint(0, arm_state.num_arms - 1)


def ucb(arm_state):
    confidence_bounds = arm_state.success_rates + np.sqrt(
        (2 * np.log(np.sum(arm_state.total_pulls + 1)))
        / np.where(arm_state.total_pulls == 0, 1, arm_state.total_pulls)
    )

    return np.argmax(confidence_bounds)


#               Temperature
# Exploitation <-----------> Exploration
def softmax(arm_state, temperature=0.1):
    exp_values = np.exp(arm_state.success_rates / temperature)
    probabilities = exp_values / np.sum(exp_values)
    return np.random.choice(len(arm_state.success_rates), p=probabilities)
