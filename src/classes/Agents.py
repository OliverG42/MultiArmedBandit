import functools
import math
from copy import deepcopy
from random import randint

import numpy as np

from Agent import Agent
from MultiArmedBandit.src.misc.temp import minimise_beta
from MultiArmedBandit.src.utils import prob_success_rate


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
    def __init__(self, name=None, alpha_prior=None, beta_prior=None):
        super().__init__()
        self._initialize()
        if name is not None:
            self.name = name
        self.num_arms = None
        self.prior_init = False

        if (alpha_prior is not None) and (beta_prior is not None):
            self.alpha_prior = alpha_prior
            self.beta_prior = beta_prior

            self.prior_init = True

            self.num_arms = len(beta_prior)

    def choose_lever(self, arm_state):
        if not self.prior_init:
            self.alpha_prior = np.ones(arm_state.num_arms)
            self.beta_prior = np.ones(arm_state.num_arms)

        samples = np.random.beta(
            arm_state.successes + self.alpha_prior + 1e-8,
            arm_state.failures + self.beta_prior + 1e-8,
        )
        return np.argmax(samples)


# Geometrically decreases the probability to do something random
def geometric_epsilon(arm_state, epsilon):
    return math.pow(epsilon, arm_state.total_pulls + 1)


class EpsilonGreedy(Agent):
    def __init__(self, name, epsilon=0.95, epsilon_function=geometric_epsilon):
        super().__init__()
        self._initialize()
        if name is None:
            self.name = f"epsilon = {epsilon}"
        else:
            self.name = name
        self.epsilon = epsilon
        self.epsilon_function = epsilon_function

    def choose_lever(self, arm_state):
        if np.random.random() < self.epsilon_function(arm_state, self.epsilon):
            # Choose at random
            chosen_arm = np.random.randint(arm_state.num_arms)
        else:
            # Choose greedily
            chosen_arm = np.argmax(arm_state.success_rates)

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


class Ripple(Agent):
    def __init__(self, the_arm_state, limit_down=0.01):
        super().__init__()
        self._initialize()
        self.name = self.name + " " + str(limit_down)
        self.num_arms = the_arm_state.num_arms
        self.limit_down = limit_down
        self.search_increment = 0.001

        self.intersection_points = [
            self._findIntersection(
                deepcopy(the_arm_state.successes[i]),
                deepcopy(the_arm_state.failures[i]),
            )
            for i in range(0, self.num_arms)
        ]
        self.arm_pulls_memory = deepcopy(the_arm_state.arm_pulls)

    # Add a dynamic cache from functools
    @functools.lru_cache(maxsize=None)
    def _findIntersection(self, successes, failures):
        upper = 1
        lower = successes / (successes + failures) if successes + failures != 0 else 1

        accuracy = 1e-4

        while upper - lower > accuracy:
            middle = (upper + lower) / 2
            success_rate = prob_success_rate(middle, successes, failures)

            if success_rate < self.limit_down:
                upper = middle
            else:
                lower = middle

        return lower

    def choose_lever(self, the_arm_state):
        # Find which intersection point(s) has changed
        for i in range(0, self.num_arms):
            if self.arm_pulls_memory[i] != the_arm_state.arm_pulls[i]:
                # Change the intersection point
                self.intersection_points[i] = self._findIntersection(
                    the_arm_state.successes[i], the_arm_state.failures[i]
                )

        result = np.argmax(self.intersection_points)

        return result


# ----------------------------------------------------------------------------------------------
# Pure exploration agents


class PureExplorationAgent(Agent):
    def do_stop(self, arm_state):
        raise NotImplementedError("The do_stop method hasn't been implemented!")

    def do_pass(self, arm_state):
        raise NotImplementedError("The do_pass method hasn't been implemented!")


class Uniform(Agent):
    def __init__(self):
        super().__init__()
        self.last_arm_index = 0

    def choose_lever(self, arm_state):
        chosen_arm = self.last_arm_index
        self.last_arm_index = (self.last_arm_index + 1) % arm_state.num_arms
        return chosen_arm

    # The Uniform Agent never stops on its own!
    def do_stop(self, arm_state):
        return False

    def get_result(self, arm_state):
        return np.argmax(arm_state.successes)


class TrackAndStop(Agent):
    def __init__(self, failure_probability=0.01):
        super().__init__()
        self._initialize()
        self.failure_probability = failure_probability
        self.previous_beta = None

    def _zt(self, arm_state):
        max_success_rate = np.max(arm_state.success_rates)
        gaps = np.array([max_success_rate - sr for sr in arm_state.success_rates])

        beta_result, min_value = minimise_beta(gaps, previous_beta=self.previous_beta)

        self.previous_beta = beta_result

        return (
                (arm_state.total_pulls + 1) * min_value / (2 * np.sum(np.exp(beta_result)))
        )

    def _bt(self, arm_state):
        k = arm_state.num_arms
        t = arm_state.total_pulls + 1
        x = 1 / math.log(self.failure_probability)
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
