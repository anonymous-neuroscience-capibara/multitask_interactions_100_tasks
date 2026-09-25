import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.cm import Reds, Blues
from typing import List, Tuple, Dict

plt.rcParams['font.size'] = 18

def compute_spike_statistics_variable(spike_counts_list: List[np.ndarray], spikes_np: np.ndarray, 
                                    epsilon: float = 1e-6, min_mean_thresh: float = 0.0001, 
                                    max_cv_thresh: float = 1000.0, max_fano_thresh: float = 1000.0) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute mean, CV, and Fano factor for true and predicted spike counts with variable-length sequences.
    
    Args:
        spike_counts_list: List of numpy arrays, each of shape (n_units, T_i) for true spikes
        spikes_np: Single numpy array of shape (n_trials, n_units, T_max) for predicted spikes
        epsilon: Small value to prevent division by zero
        min_mean_thresh: Minimum mean firing rate threshold
        max_cv_thresh: Maximum coefficient of variation threshold
        max_fano_thresh: Maximum Fano factor threshold
    
    Returns:
        Dictionary with masked statistics (outlier units removed)
    """
    n_trials = len(spike_counts_list)
    n_units_true = spike_counts_list[0].shape[0]  # Number of units in true spikes (first dimension)
    n_units_pred = spikes_np.shape[1]  # Number of units in predicted spikes (second dimension)
    
    print(f"True spikes: {n_units_true} units")
    print(f"Predicted spikes: {n_units_pred} units")
    
    if n_units_true != n_units_pred:
        print(f"Warning: Mismatched number of units between true ({n_units_true}) and predicted ({n_units_pred}) spikes")
        print(f"Using only the first {n_units_pred} units from true spikes for comparison")
        # Truncate true spikes to match predicted units
        spike_counts_list = [seq[:n_units_pred, :] for seq in spike_counts_list]
        n_units = n_units_pred
    else:
        n_units = n_units_true
    
    # Initialize arrays to store statistics
    true_means = np.zeros((n_trials, n_units))
    pred_means = np.zeros((n_trials, n_units))
    true_stds = np.zeros((n_trials, n_units))
    pred_stds = np.zeros((n_trials, n_units))
    true_counts = np.zeros((n_trials, n_units))
    pred_counts = np.zeros((n_trials, n_units))
    
    # Compute statistics for each trial
    for i in range(n_trials):
        # Get the actual length of this trial
        true_seq = spike_counts_list[i]  # (n_units, T_i)
        T_i = true_seq.shape[1]  # Get actual length of this trial
        pred_seq = spikes_np[i, :, :T_i]  # (n_units, T_i) - truncate predicted to match true length
        
        # Verify shapes before computing statistics
        if true_seq.shape != pred_seq.shape:
            raise ValueError(f"Shape mismatch in trial {i}: true_seq {true_seq.shape} vs pred_seq {pred_seq.shape}")
        
        # Mean and std across time for this trial
        true_means[i] = true_seq.mean(axis=1)  # Mean across time for each unit
        pred_means[i] = pred_seq.mean(axis=1)
        true_stds[i] = true_seq.std(axis=1)
        pred_stds[i] = pred_seq.std(axis=1)
        
        # Total spike counts for this trial
        true_counts[i] = true_seq.sum(axis=1)
        pred_counts[i] = pred_seq.sum(axis=1)
    
    # Compute mean firing rates across trials
    true_mean = true_means.mean(axis=0)
    pred_mean = pred_means.mean(axis=0)
    
    # Compute CV (using mean of stds across trials)
    true_cv = true_stds.mean(axis=0) / (true_mean + epsilon)
    pred_cv = pred_stds.mean(axis=0) / (pred_mean + epsilon)
    
    # Compute Fano factor (variance of counts across trials / mean of counts)
    true_fano = true_counts.var(axis=0) / (true_mean + epsilon)
    pred_fano = pred_counts.var(axis=0) / (pred_mean + epsilon)
    
    #print(true_mean)
    #print(pred_mean)
    #print(true_cv)
    #print(pred_cv)
    # Filter out outlier units

    valid_units = (
        (true_mean > min_mean_thresh) &
        (pred_mean > min_mean_thresh) &
        (true_cv < max_cv_thresh) &
        (pred_cv < max_cv_thresh) &
        (true_fano < max_fano_thresh) &
        (pred_fano < max_fano_thresh)
    )
    
    print(f"Number of valid units after filtering: {valid_units.sum()}")
    
    return {
        "mean": (true_mean[valid_units], pred_mean[valid_units]),
        "cv": (true_cv[valid_units], pred_cv[valid_units]),
        "fano": (true_fano[valid_units], pred_fano[valid_units])
    }

def compute_spike_statistics(spike_counts, spikes_np, epsilon=1e-6, min_mean_thresh=0.01, max_cv_thresh=10.0, max_fano_thresh=10.0):
    """
    Original version for fixed-length sequences.
    Compute mean, CV, and Fano factor for true and predicted spike counts.
    Returns masked statistics (outlier units removed).
    """
    n_trials, n_bins, n_units = spike_counts.shape
    truth = np.transpose(spike_counts, (0, 2, 1))
    # Transpose predicted to match true shape
    pred = np.transpose(spikes_np, (0, 2, 1))  # (n_trials, n_units, n_bins)
    # Mean and std across time (flattened)
    true_flat = truth.transpose(1, 0, 2).reshape(n_units, -1)
    pred_flat = pred.transpose(1, 0, 2).reshape(n_units, -1)
    
    true_mean_time = true_flat.mean(axis=1)
    pred_mean_time = pred_flat.mean(axis=1)
    true_std = true_flat.std(axis=1)
    pred_std = pred_flat.std(axis=1)
    true_cv = true_std / (true_mean_time + epsilon)
    pred_cv = pred_std / (pred_mean_time + epsilon)
    
    # Fano factor across trials (summed over time)
    true_counts = spike_counts.sum(axis=1)  
    pred_counts = spikes_np.sum(axis=1)
    true_mean = true_counts.mean(axis=0)
    pred_mean = pred_counts.mean(axis=0)
    true_var = true_counts.var(axis=0)
    pred_var = pred_counts.var(axis=0)
    true_fano = true_var / (true_mean + epsilon)
    pred_fano = pred_var / (pred_mean + epsilon)

    
    valid_units = (
        (true_mean > min_mean_thresh) &
        (pred_mean > min_mean_thresh) &
        (true_cv < max_cv_thresh) &
        (pred_cv < max_cv_thresh) &
        (true_fano < max_fano_thresh) &
        (pred_fano < max_fano_thresh)
    )
    
    return {
        "mean": (true_mean[valid_units], pred_mean[valid_units]),
        "cv": (true_cv[valid_units], pred_cv[valid_units]),
        "fano": (true_fano[valid_units], pred_fano[valid_units])
    }



def multiplot_statistics(stat_dict, which=["mean", "cv", "fano"], bin_size=0.025):
    """
    Creates scatter plots for selected statistics: mean, CV, and/or Fano.
    
    Args:
        stat_dict: dict with keys "mean", "cv", "fano" mapping to (true, pred) tuples
        which: list of statistics to plot, any subset of ["mean", "cv", "fano"]
        bin_size: bin size in seconds (used to convert mean firing rates to Hz)
    """
    labels = {
        "mean": ("Mean Firing Rate (Hz)", "True Mean (Hz)", "Predicted Mean (Hz)"),
        "cv":   ("Coefficient of Variation", "True CV", "Predicted CV"),
        "fano": ("Fano Factor", "True Fano", "Predicted Fano"),
    }

    n_plots = len(which)
    fig, axs = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    if n_plots == 1:
        axs = [axs]  # make it iterable

    for i, key in enumerate(which):
        true_vals, pred_vals = stat_dict[key]

        if key == "mean":
            true_vals = true_vals / bin_size
            pred_vals = pred_vals / bin_size

        r, _ = pearsonr(true_vals, pred_vals)
        axs[i].scatter(true_vals, pred_vals, color=Reds(0.8), label=f"r = {r:.3f}")
        lims = [min(true_vals.min(), pred_vals.min()), max(true_vals.max(), pred_vals.max())]
        axs[i].plot(lims, lims, '--', color=Blues(0.8), alpha=0.7)
       # axs[i].set_title(labels[key][0])
        axs[i].set_xlabel(labels[key][1])
        axs[i].set_ylabel(labels[key][2])
        axs[i].legend()

    plt.tight_layout()
    plt.show()