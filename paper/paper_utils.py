import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_failure_rates_smooth(T, results, window_size=3, color='red', label='Smooth estimate'):
    """
    Plot failure rates using smooth estimation methods.

    Parameters:
    T: array-like of T values (continuous)
    results: array-like of binary results (0 for failure, 1 for success)
    window_size: bandwidth for smoothing
    method: 'rolling', 'kde', or 'both' for comparison
    """
    # Create evaluation points
    T_eval = np.linspace(min(T), max(T), 500)

    # Plot raw data points
    # plt.scatter(T, results, alpha=0.1, color=color, label='Raw data')

    recall = []
    ci_upper = []
    ci_lower = []
    for t in T_eval:
        # Calculate weights based on distance
        weights = np.exp(-(T - t) ** 2 / (2 * window_size ** 2))
        weights = weights / np.sum(weights)

        # Weighted mean
        weighted_rate = np.sum(weights * results)

        # Weighted variance for confidence interval
        n_effective = 1 / np.sum(weights ** 2)  # Effective sample size
        weighted_var = np.sum(weights * (results - weighted_rate) ** 2) * (n_effective / (n_effective - 1))
        ci = 1.96 * np.sqrt(weighted_var ** 2 / n_effective)

        recall.append(weighted_rate)
        ci_upper.append(min(1, weighted_rate + ci))
        ci_lower.append(max(0, weighted_rate - ci))

    plt.plot(T_eval, recall, '-', linewidth=2, label=label, color=color)
    plt.fill_between(T_eval, ci_lower, ci_upper, color=color, alpha=0.2)


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    T = np.random.uniform(0, 10, 1000)
    probability_of_failure = 1 / (1 + np.exp(-(T - 5)))  # Sigmoid function
    results = np.random.random(len(T)) > probability_of_failure

    # Create plots with different methods
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plot_failure_rates_smooth(T, results, method='rolling')
    plt.title('Rolling Window Method')

    plt.subplot(132)
    plot_failure_rates_smooth(T, results, method='kde')
    plt.title('KDE Method')

    plt.subplot(133)
    plot_failure_rates_smooth(T, results, method='both')
    plt.title('Both Methods Compared')

    plt.tight_layout()
    plt.show()
