import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

matplotlib_palette = sns.color_palette("tab10")
muted_palette = sns.color_palette("dark")
COLORS = {
    'crai': matplotlib_palette[4],
    'crai_fitmap': matplotlib_palette[0],
    'dockim': matplotlib_palette[3],
    'uy': muted_palette[4],
    'uy_fitmap': muted_palette[0],
}

LABELS = {
    'gt': r'\texttt{Ground Truth}',
    'crai': r'\texttt{CrAI}',
    'crai_fitmap':r'\texttt{CrAI FitMap}',
    'dockim': r'\texttt{dock in map}',
    'uy': r'$\overrightarrow{u_y}$',
    'uy_fitmap':'$\overrightarrow{u_y} FitMap$',
}


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
    pass
