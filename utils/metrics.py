import numpy as np
from scipy.stats import pearsonr


def compute_mae(predictions, targets):
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.array(predictions) - np.array(targets))))


def compute_rmse(predictions, targets):
    """Root Mean Square Error."""
    return float(np.sqrt(np.mean((np.array(predictions) - np.array(targets)) ** 2)))


def compute_pearson(predictions, targets):
    """Pearson Correlation Coefficient."""
    if len(predictions) < 2:
        return 0.0
    corr, _ = pearsonr(predictions, targets)
    return float(corr)


def compute_mape(predictions, targets):
    """Mean Absolute Percentage Error."""
    targets = np.array(targets, dtype=float)
    predictions = np.array(predictions, dtype=float)
    mask = targets != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100)


def compute_all_metrics(predictions, targets):
    """Compute all metrics."""
    return {
        "MAE": round(compute_mae(predictions, targets), 4),
        "RMSE": round(compute_rmse(predictions, targets), 4),
        "Pearson": round(compute_pearson(predictions, targets), 4),
        "MAPE": round(compute_mape(predictions, targets), 4),
    }


def print_metrics(metrics, title="Evaluation Metrics"):
    """Pretty print metrics."""
    print(f"\n{'=' * 40}")
    print(f"  {title}")
    print(f"{'=' * 40}")
    for key, value in metrics.items():
        print(f"  {key:>10}: {value}")
    print(f"{'=' * 40}\n")