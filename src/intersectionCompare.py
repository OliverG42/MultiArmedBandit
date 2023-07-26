import sys
import timeit

import numpy as np

from src.experimental import probSuccessRate

limit_down = 0.1

global prob_calls


# @functools.lru_cache(maxsize=None)
# No caching allowed!
def V1(successes, failures):
    accuracy = 1e-3

    arm_value = 1

    while arm_value >= 0 and successes + failures != 0:
        success_rate = probSuccessRate(arm_value, successes, failures)

        global prob_calls
        prob_calls += 1

        if success_rate > limit_down:
            return arm_value

        arm_value -= accuracy

    return 1


# @functools.lru_cache(maxsize=None)
# No caching allowed!
def V2(successes, failures):
    upper = 1
    lower = successes / (successes + failures) if successes + failures != 0 else 1

    accuracy = 1e-3

    while upper - lower > accuracy:
        middle = (upper + lower) / 2
        success_rate = probSuccessRate(middle, successes, failures)

        global prob_calls
        prob_calls += 1

        if success_rate < limit_down:
            upper = middle
        else:
            lower = middle

    return lower


class Suite:
    def __init__(self, name, lower, upper, jump=1):
        self.name = name
        self.values = [(i, j) for i in range(lower, upper, jump) for j in range(lower, upper, jump)]


if __name__ == "__main__":

    functions = [V1, V2]

    suites = [
        Suite("First few pulls", 0, 10),
        Suite("After 100 pulls", 100, 110),
        Suite("Near 1000 pulls", 1000, 1010),
        Suite("One large range", 100, 400, jump=30),
        Suite("Quite a big dif", 500, 1000, jump=50),
        Suite("Some huge jumps", 1000, 2000, jump=100),
        Suite("In the long run", 0, 2000, jump=200),
    ]

    for suite in suites:

        results = []

        for function in functions:

            prob_calls = 0

            def perform_suite():
                outcomes = []
                for (successes, failures) in suite.values:
                    outcomes.append(function(successes, failures))
                return outcomes


            execution_time = timeit.timeit(perform_suite, number=1)

            results.append(perform_suite())

            print(f"Time for {function.__name__} in '{suite.name}': {execution_time:.5f}s. Called probSuccessRate {prob_calls} times")

        # Define the tolerance (atol) for the closeness comparison
        tolerance = 1e-2

        # Check if elements are not close
        not_close_mask = ~np.isclose(results[0], results[1], atol=tolerance)
        not_close_indices = np.where(not_close_mask)[0]

        if not not_close_indices.size:
            # Yay!
            pass
        else:
            print("Warning - functions are giving too different answers!")
            for index in not_close_indices:
                print(f"{results[0][index]} is not close enough to {results[1][index]}")

            sys.exit()
