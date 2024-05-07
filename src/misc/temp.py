import math
import numpy as np
from scipy.optimize import basinhopping
import matplotlib

matplotlib.use("TkAgg")  # Switch backend to TkAgg
import matplotlib.pyplot as plt


def objective_function(beta, gap_values):
    beta_max = max(beta)
    min_value = np.inf

    # Cases where all arms are identical should give a value of 0
    if np.all(gap_values == 0):
        return 0

    for i in range(0, len(beta)):
        # Ignore the cases where the arm is the "best" arm
        if gap_values[i] == 0:
            continue

        current_value = (
            (gap_values[i] ** 2)
            / (np.sum(np.exp(beta)))
            * (np.exp(beta_max) * np.exp(beta[i]))
            / (np.exp(beta_max) + np.exp(beta[i]))
        )
        if current_value < min_value:
            min_value = current_value

    return min_value


def log_objective_function(beta, gap_values, penalty_multiplier=0.1):
    beta_max = np.max(beta)

    # Replace zero values in gap_values with a very small positive number, to avoid division by zero
    gap_values = np.where(gap_values == 0, 1000, gap_values)

    exp_beta = np.exp(-beta)
    exp_beta_max = np.exp(-beta_max)

    log_gap_values = np.log(gap_values)
    log_term = 2 * log_gap_values - np.log(exp_beta + exp_beta_max)

    # Penalise very large values of beta
    penalty = np.sum(beta**2) * penalty_multiplier

    result = -np.log(2) - np.log(np.sum(np.exp(beta))) + np.min(log_term) - penalty

    # Since we're maximising the minimum, return the negative
    return -result


# Returns a tuple containing the betas resulting in the minimum value of the LOG OBJECTIVE FUNCTION
# and the minimum value when using these betas in the LOG OBJECTIVE FUNCTION.
def get_minimising_beta_data(gap_values, previous_beta=None, iterations=5):
    if previous_beta is None:
        previous_beta = np.array(
            [np.random.uniform(-5, 5) for _ in range(len(gap_values))]
        )
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

    constraints = [*not_too_small, *not_too_big]

    previous_beta_noisy = previous_beta + np.random.normal(
        loc=0, scale=0.1, size=len(previous_beta)
    )

    minimizer_kwargs = {
        "args": (gap_values,),
        "bounds": [(None, None) for _ in range(len(previous_beta))],
        "constraints": constraints,
        "tol": 1e-5,
    }

    # Define the basin hopping optimizer
    optimizer = basinhopping(
        log_objective_function,
        previous_beta_noisy,
        minimizer_kwargs=minimizer_kwargs,
        niter=iterations,
        stepsize=0.5,
    )

    return optimizer.x, optimizer.fun


# Returns the best beta values, and the minimum respective value of the OBJECTIVE FUNCTION
def minimise_beta(gap_values, previous_beta=None):
    beta_result, log_function_min_value = get_minimising_beta_data(
        gap_values, previous_beta=previous_beta
    )
    min_value = objective_function(beta_result, gap_values)
    return beta_result, min_value


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

    min_value = objective_function(beta_result, gap_values)

    print("Optimal solution found!")
    print(f"Beta values: {[round(beta, 4) for beta in beta_result]}")
    print(
        f"Respective alpha values: {[round(np.exp(beta) / np.sum(np.exp(beta_result)), 4) for beta in beta_result]}"
    )
    print(f"Minimum value of the objective function: {min_value}")
