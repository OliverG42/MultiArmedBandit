from collections import namedtuple
from random import randint

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem

ArmState = namedtuple("ArmState", ["successes", "failures", "num_arms"])


def getSuccessRates(arm_state, divisionByZeroIs=0):
    total_trials = arm_state.successes + arm_state.failures

    success_rates = arm_state.successes / total_trials.size

    # Force arms that haven't been tried yet to have a success rate of 1
    success_rates[(success_rates == 0.0) & (total_trials == 0)] = divisionByZeroIs

    return success_rates


def bernNearlyGreedy(arm_state):
    # Avoids messy division by zero warning
    theta = getSuccessRates(arm_state, divisionByZeroIs=1)
    return np.argmax(theta)


def bernGreedy(arm_state):
    # Avoids messy division by zero warning
    theta = getSuccessRates(arm_state)
    return np.argmax(theta)


def bernTS(arm_state):
    samples = np.random.beta(arm_state.successes + 1, arm_state.failures + 1)
    return np.argmax(samples)


def epsilonGreedy(arm_state, epsilon=0.1):
    if np.random.random() < epsilon:
        # Choose at random
        chosen_arm = np.random.randint(arm_state.num_arms)
    else:
        # Choose greedily
        success_rates = getSuccessRates(arm_state, divisionByZeroIs=0)
        chosen_arm = np.argmax(success_rates)

    return chosen_arm


def completelyRandom(arm_state):
    return randint(0, arm_state.num_arms - 1)


def ucb(arm_state):
    total_trials = arm_state.successes + arm_state.failures
    success_rates = getSuccessRates(arm_state, divisionByZeroIs=1)

    confidence_bounds = success_rates + np.sqrt(
        (2 * np.log(np.sum(total_trials + 1))) / np.where(total_trials == 0, 1, total_trials)
    )

    return np.argmax(confidence_bounds)


#               Temperature
# Exploitation <-----------> Exploration
def softmax(arm_state, temperature=0.1):
    success_rates = getSuccessRates(arm_state, divisionByZeroIs=1)

    exp_values = np.exp(success_rates / temperature)
    probabilities = exp_values / np.sum(exp_values)
    return np.random.choice(len(success_rates), p=probabilities)


def calculate_regret(reward_probs, chosen_arm):
    max_reward_prob = np.max(reward_probs)
    chosen_reward_prob = reward_probs[chosen_arm]
    regret = max_reward_prob - chosen_reward_prob
    return regret


def perform_sampling(choosing_function, reward_probs, num_trials):
    num_arms = len(reward_probs)
    arm_state = ArmState(
        successes=np.zeros(num_arms), failures=np.zeros(num_arms), num_arms=num_arms
    )
    allRegrets = []

    for _ in range(num_trials):
        chosen_arm = choosing_function(arm_state)
        reward = np.random.binomial(1, reward_probs[chosen_arm])

        if reward:
            arm_state.successes[chosen_arm] += 1
        else:
            arm_state.failures[chosen_arm] += 1

        regret = calculate_regret(reward_probs, chosen_arm)
        allRegrets.append(regret)

    return allRegrets


def makeGraph(reward_probs):
    print(reward_probs)
    num_trials = 100  # How many "lever pulls" are there?
    num_samples = 100  # How many "runs" are there?

    num_trials += 1
    errorBarInterval = int(num_trials / 10)

    plt.figure(figsize=(10, 6))

    functions = [
        bernTS,
        bernGreedy,
        bernNearlyGreedy,
        completelyRandom,
        epsilonGreedy,
        ucb,
        softmax,
    ]
    colours = ["red", "yellow", "green", "blue", "purple", "pink", "brown"]

    for choosing_function, colour in zip(functions, colours):
        print("Computing for", choosing_function.__name__)
        all_regrets = []
        for _ in range(num_samples):
            regrets = perform_sampling(choosing_function, reward_probs, num_trials)
            cumulative_regrets = np.cumsum(regrets)
            all_regrets.append(cumulative_regrets)

            # Plotting the cumulative regret with adjusted line width
            plt.plot(
                range(1, num_trials + 1),
                cumulative_regrets,
                alpha=0.05,
                linewidth=0.8,
                color=colour,
            )

        avg_regrets = np.mean(all_regrets, axis=0)

        # Plot the smooth curve of average cumulative regrets
        plt.plot(
            range(1, num_trials + 1, errorBarInterval),
            avg_regrets[::errorBarInterval],
            label=f"{choosing_function.__name__}",
            linewidth=2,
            color=colour,
        )

    plt.xlabel("Trials" f"{reward_probs}")
    plt.ylabel("Cumulative Regret")
    plt.legend()


if __name__ == "__main__":
    makeGraph([0.6, 0.65, 0.7])
    plt.show()
