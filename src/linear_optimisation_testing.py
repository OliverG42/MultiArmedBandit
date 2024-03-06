import math
import numpy as np
from scipy.optimize import minimize
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


def log_objective_function(beta, gap_values, do_penalty=True):
    beta_max = max(beta)
    min_value = np.inf

    # Cases where all arms are identical should give a value of 0
    if np.all(gap_values == 0):
        return 0

    for i in range(0, len(beta)):
        if gap_values[i] == 0:
            continue
        else:
            current_value = 2 * math.log(gap_values[i]) - math.log(
                    np.exp(-beta[i]) + np.exp(-beta_max)
                )
        if current_value < min_value:
            min_value = current_value

    # Penalise very large values of beta
    if do_penalty:
        penalty = np.sum(pow(beta, 2)) * 0.1
    else:
        penalty = 0

    return (-math.log(np.sum(np.exp(beta))) + min_value) + penalty


# Returns a tuple containing the betas resulting in the minimum value of the LOG OBJECTIVE FUNCTION
# and the minimum value when using these betas in the OBJECTIVE FUNCTION.
def get_minimising_beta_data(gap_values, beta_prior=None, iterations=10):
    if beta_prior is None:
        beta_prior = np.array(
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
    results = []
    min_values = []

    for _ in range(1, iterations):
        beta_prior_noisy = beta_prior + np.random.normal(loc=0, scale=0.5, size=len(beta_prior))
        res = minimize(
            log_objective_function,
            # Add some random noise to the beta_prior
            beta_prior_noisy,
            gap_values,
            bounds=[(None, None) for _ in range(len(beta_prior))],
            constraints=constraints,
            tol=1e-5,
        )
        if res.success:
            results.append(res.x)
            min_values.append(res.fun)
        else:
            print("Optimization failed. Check constraints or initial values.")
            print(f"Failed with:\nbeta_prior={list(beta_prior)}")
            print(f"gap_values={list(gap_values)}")
            exit(0)
    return results, min_values


# Returns the best beta values, and the minimum respective value of the OBJECTIVE FUNCTION
def minimise_beta(gap_values, beta_prior=None):
    beta_results, min_values = get_minimising_beta_data(gap_values, beta_prior=beta_prior)
    return determine_result(beta_results, min_values, gap_values)


# Returns the best beta values, and the minimum respective value of the OBJECTIVE FUNCTION
def determine_result(all_beta_results, min_values, gap_values):
    all_beta_results = np.array(all_beta_results)
    min_values = np.array(min_values)

    if USE_MIN:
        absolute_min_index = np.argmin(min_values)
        beta_result = all_beta_results[absolute_min_index]
    else:
        # Sort the indices based on the minimum values obtained
        sorted_indices = np.argsort(min_values)

        # Determine the number of results to consider for averaging (top 10%)
        num_results = len(all_beta_results)
        num_top_results = max(1, int(0.1 * num_results))

        # Select the top 10% of results based on the sorted indices
        top_results = all_beta_results[sorted_indices[:num_top_results]]

        # Calculate the average of the selected results
        beta_result = np.mean(top_results, axis=0)

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


# Determine if you want the best result to be defined by the absolute min or the average of the best 10%
USE_MIN = True

if __name__ == "__main__":
    success_rates = np.array([0.01, 0.02, 0.03, 0.05, 0.5, 0.51])
    gap_values = np.array([np.max(success_rates) - sr for sr in success_rates])

    beta_results, min_values = get_minimising_beta_data(gap_values, beta_prior=None)
    beta_result, min_value = determine_result(beta_results, min_values, gap_values)
    plot_beta_values(beta_results)

    print("Optimal solution found!")
    print(f"Beta values: {[round(beta, 4) for beta in beta_result]}")
    print(
        f"Respective alpha values: {[round(np.exp(beta) / np.sum(np.exp(beta_result)), 4) for beta in beta_result]}"
    )
    print(f"Minimum value of the objective function: {min_value}")
