import timeit
import numpy as np


# Original version of my lazy_integration function
def lazy_integration_original(function, *args, step_size=0.001):
    return step_size * sum(
        float(function(x, *args))
        for x in [step_size * i for i in range(int(1 / step_size) + 1)]
    )


# Optimized using numpy vectorization
def lazy_integration_vectorized(function, *args, step_size=0.001):
    x_values = np.arange(0, 1 + step_size, step_size)
    result = np.sum(function(x_values, *args)) * step_size
    return result


# TODO Run this on computer with scipy
"""from scipy.integrate import quad

def lazy_integration(function, *args):
    result, _ = quad(function, 0, 1, args=args)
    return result"""


# A testing function - square
def square_function(x):
    return x ** 2


if __name__ == "__main__":
    intensity = 2000
    is_close_error = 1e-3

    integration_methods = [
        ("Original", lazy_integration_original),
        ("Vectorized", lazy_integration_vectorized),
    ]

    integration_results = []

    for name, integration_method in integration_methods:
        start_time = timeit.timeit(lambda: integration_method(square_function), number=intensity)
        result = integration_method(square_function)

        print(f"Time taken for integration using {name} method: {start_time:.8f} seconds")

        integration_results.append((name, result))

    # Custom function to check if the results are not close to each other
    # Since np.isclose was being difficult and throwing errors due to integration_results' structure
    def are_results_close(results, rtol):
        reference_result = results[0][1]
        return all(abs(a_result - reference_result) <= rtol for (_, a_result) in results)


    # Check if the results are not close to each other
    if not are_results_close(integration_results, is_close_error):
        print("Results are not close to each other:")
        [print(f"{name}: {result}") for (name, result) in integration_results]
