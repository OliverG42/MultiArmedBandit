from random import randint

import numpy as np


class Agent:
    def __init__(self):
        self.name = self.__class__.__name__
        self._initialized = False

    def choose_lever(self, arm_state):
        raise NotImplementedError("The chooseLever method hasn't been implemented!")

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._init_args = args
        instance._init_kwargs = kwargs
        return instance

    def reset(self):
        if self._initialized:
            # noinspection PyArgumentList
            self.__init__(*self._init_args, **self._init_kwargs)

    def _initialize(self):
        self._initialized = True


class BernGreedy(Agent):
    def __init__(self, num_levers, burn_time=10):
        super().__init__()
        self._initialize()
        self.num_levers = num_levers
        self.burn_time = num_levers * burn_time

    def choose_lever(self, arm_state):
        # Choose fairly
        if self.burn_time > 0:
            self.burn_time -= 1
            return self.burn_time % self.num_levers
        # Choose greedily
        else:
            return np.argmax(arm_state.success_rates)


class BernTS(Agent):
    def choose_lever(self, arm_state, alpha_prior=1, beta_prior=1):
        samples = np.random.beta(
            arm_state.successes + alpha_prior, arm_state.failures + beta_prior
        )
        return np.argmax(samples)


class EpsilonGreedy(Agent):
    def __init__(self, epsilon=0.95):
        super().__init__()
        self._initialize()
        self.epsilon = epsilon
        self.CONST_epsilon = epsilon

    def choose_lever(self, arm_state):
        if np.random.random() < self.epsilon:
            # Choose at random
            chosen_arm = np.random.randint(arm_state.num_arms)
        else:
            # Choose greedily
            chosen_arm = np.argmax(arm_state.success_rates)

        # Slowly decrease the probability to do something random
        self.epsilon *= self.CONST_epsilon

        return chosen_arm


class CompletelyRandom(Agent):
    def choose_lever(self, arm_state):
        return randint(0, arm_state.num_arms - 1)


class Ucb(Agent):
    def choose_lever(self, arm_state):
        confidence_bounds = arm_state.success_rates + np.sqrt(
            (2 * np.log(np.sum(arm_state.total_pulls + 1)))
            / np.where(arm_state.total_pulls == 0, 1, arm_state.total_pulls)
        )

        return np.argmax(confidence_bounds)


#               Temperature
# Exploitation <-----------> Exploration
class Softmax(Agent):
    def __init__(self, temperature=0.1):
        super().__init__()
        self._initialize()
        self.temperature = temperature

    def choose_lever(self, arm_state):
        exp_values = np.exp(arm_state.success_rates / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        return np.random.choice(len(arm_state.success_rates), p=probabilities)
