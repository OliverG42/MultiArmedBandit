# @title Default title text
import math
import numpy as np
from scipy.optimize import basinhopping
import matplotlib

matplotlib.use("TkAgg")  # Switch backend to TkAgg
import matplotlib.pyplot as plt


def minimising_section(beta, gap_values, penalty=0.1):
    beta_max = max(beta)
    min_value = np.inf

    # Cases where all arms are identical should give a value of 0
    if np.all(gap_values == 0):
        return 0

    for i in range(0, len(beta)):
        # Ignore the cases where the arm is the "best" arm
        if gap_values[i] == 0:
            continue

        current_value = (gap_values[i] ** 2) / (np.exp(-beta[i]) + np.exp(-beta_max))

        if current_value < min_value:
            min_value = current_value

    # Penalise very large values of beta
    return min_value + np.sum(beta**2) * penalty


def get_constraints(gap_values):
    average_min_value = -200
    not_too_small = [
        {
            "type": "ineq",
            "fun": lambda beta, i=i, min_val=average_min_value: beta[i] - min_val,
        }
        for i in range(len(gap_values))
    ]

    max_value = 200
    not_too_big = [
        {"type": "ineq", "fun": lambda beta, i=i, max_val=max_value: max_val - beta[i]}
        for i in range(len(gap_values))
    ]

    return [*not_too_small, *not_too_big]


def get_minimising_beta_data(
    gap_values, previous_beta=None, iterations=10, function=minimising_section
):
    if previous_beta is None:
        previous_beta = np.array(
            [np.random.uniform(-5, 5) for _ in range(len(gap_values))]
        )

    previous_beta_noisy = previous_beta + np.random.normal(
        loc=0, scale=0.1, size=len(previous_beta)
    )

    constraints = get_constraints(gap_values)

    minimizer_kwargs = {
        "args": (gap_values,),
        "bounds": [(None, None) for _ in range(len(previous_beta))],
        "constraints": constraints,
        "tol": 1e-5,
    }

    optimizer = basinhopping(
        function,
        previous_beta_noisy,
        minimizer_kwargs=minimizer_kwargs,
        niter=iterations,
        stepsize=0.5,
    )

    return optimizer.x, optimizer.fun


# Returns the best beta values, and the minimum respective value
def minimise_beta(gap_values, previous_beta=None):
    beta_result, _ = get_minimising_beta_data(gap_values, previous_beta=previous_beta)
    clean_min_value = minimising_section(beta_result, gap_values, penalty=0)
    return beta_result, clean_min_value


def plot_beta_values(results):
    results = np.array(results)
    for i, beta_values in enumerate(results.T):
        jittered_x = np.random.normal(i, 0.1, size=len(beta_values))
        plt.scatter(jittered_x, beta_values, label=f"Beta {i + 1}", alpha=0.5, s=5)
    plt.xlabel("Beta Index")
    plt.ylabel("Beta Values")
    plt.title("Scatter Plot of Beta Values")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    success_rates = np.array([0.01, 0.02, 0.03, 0.05, 0.5, 0.51])
    gap_values = np.array([np.max(success_rates) - sr for sr in success_rates])

    beta_result, log_function_min_value = get_minimising_beta_data(
        gap_values, previous_beta=None
    )

    min_value = minimising_section(beta_result, gap_values)

    print("Optimal solution found!")
    print(f"Beta values: {[round(beta, 4) for beta in beta_result]}")
    print(
        f"Respective alpha values: {[round(np.exp(beta) / np.sum(np.exp(beta_result)), 4) for beta in beta_result]}"
    )
    print(f"Minimum value of the objective function: {min_value}")
