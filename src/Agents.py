import math
from random import randint

import numpy as np

from linear_optimisation_testing import minimise_beta, get_minimising_beta_data, determine_result, \
    log_objective_function


class Agent:
    def __init__(self):
        self.name = self.__class__.__name__
        self._initialized = False

    def choose_lever(self, arm_state):
        raise NotImplementedError("The choose_lever method hasn't been implemented!")

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
        exploitation_factor = arm_state.success_rates

        exploration_factor = (
                1
                / 2
                * np.sqrt(
            (np.log(arm_state.total_pulls + 1))
            / np.where(
                arm_state.arm_pulls == 0,
                np.ones(arm_state.num_arms),
                arm_state.arm_pulls,
            )
        )
        )

        confidence_bounds = exploitation_factor + exploration_factor

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


# ----------------------------------------------------------------------------------------------
# Pure exploration agents


class PureExplorationAgent(Agent):
    def do_stop(self, arm_state):
        raise NotImplementedError("The do_stop method hasn't been implemented!")

    def do_pass(self, arm_state):
        raise NotImplementedError("The do_pass method hasn't been implemented!")


class Uniform(Agent):
    def choose_lever(self, arm_state):
        return np.argmax(arm_state.success_rates)

    # The Uniform Agent never stops on its own!
    def do_stop(self, arm_state):
        return False

    def get_result(self, arm_state):
        return np.argmax(arm_state.successes)


class TrackAndStop(Agent):
    def __init__(self, failure_probability=0.1):
        super().__init__()
        self._initialize()
        self.failure_probability = failure_probability
        self.previous_beta = None

    def _zt(self, arm_state):
        max_success_rate = np.max(arm_state.success_rates)
        gaps = np.array([max_success_rate - sr for sr in arm_state.success_rates])

        beta_result, min_value = minimise_beta(gaps, beta_prior=self.previous_beta)

        self.previous_beta = beta_result

        exponent = -math.log(2) + log_objective_function(beta_result, gaps, do_penalty=False)

        print(f"constant: {math.log(2)}")
        print(f"log_function: {log_objective_function(beta_result, gaps, do_penalty=False)}")
        print(f"exponent: {exponent}")
        print(f"returns: {0.5 * np.exp(exponent)}\n")

        return 0.5 * np.exp(exponent)

    def _bt(self, arm_state):
        k = arm_state.num_arms
        t = arm_state.total_pulls + 1
        # TODO Correct value for x?
        x = 1/math.log(self.failure_probability)
        return k * math.log(t * (t + 1)) + x

    def choose_lever(self, arm_state):
        # Checks also if each arm has been pulled at least once
        t = arm_state.total_pulls
        least_picked_arm = np.argmin(arm_state.arm_pulls)

        if arm_state.arm_pulls[least_picked_arm] < math.sqrt(t):
            return least_picked_arm
        else:
            return np.argmax((t * arm_state.success_rates) - arm_state.arm_pulls)

    def do_stop(self, arm_state):
        if self.previous_beta is None:
            self.previous_beta = np.random.normal(
                loc=0, scale=5, size=arm_state.num_arms
            )
        zt = self._zt(arm_state)
        bt = self._bt(arm_state)

        print(f"zt = {zt}")
        print(f"bt = {bt}")

        return zt >= bt

    def do_pass(self, arm_state):
        return np.argmax(arm_state.successes)

    def get_result(self, arm_state):
        return np.argmax(arm_state.successes)
