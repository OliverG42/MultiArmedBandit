import numpy as np

class ArmState:
    def __init__(self, reward_probs):
        self.reward_probs = reward_probs
        self.max_prob = np.max(reward_probs)
        self.num_arms = len(reward_probs)
        self.successes = np.zeros(self.num_arms)
        self.failures = np.zeros(self.num_arms)
        self.arm_pulls = np.zeros(self.num_arms)
        self.total_pulls = 0
        self.success_rates = np.ones(self.num_arms)
        self.regrets = []

    def pull_arm(self, arm_number, force_result=None):
        if force_result is None:
            outcome = np.random.binomial(1, self.reward_probs[arm_number])
        else:
            outcome = force_result
        if outcome:
            self.successes[arm_number] += 1
        else:
            self.failures[arm_number] += 1

        self.arm_pulls[arm_number] += 1
        self.total_pulls += 1

        # Update success rates
        self.success_rates[arm_number] = round(
            self.successes[arm_number] / self.arm_pulls[arm_number], 4
        )

        # Update regret
        self.regrets.append(round(self.max_prob - self.reward_probs[arm_number], 4))