from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


def align_bitcode_distributions(distributions_dict):
    # Get all unique bitcodes across all tasks
    all_bitcodes = set()
    for dist in distributions_dict.values():
        all_bitcodes.update(dist.keys())

    # Sort bitcodes for consistent ordering
    bitcode_labels = sorted(list(all_bitcodes))
    n_bitcodes = len(bitcode_labels)
    n_tasks = len(distributions_dict)

    # Create aligned matrix
    aligned_matrix = np.zeros((n_tasks, n_bitcodes))
    task_id_to_idx = {
        task_id: idx for idx, task_id in enumerate(distributions_dict.keys())
    }
    for task_id, dist in distributions_dict.items():
        task_idx = task_id_to_idx[task_id]
        for i, bitcode in enumerate(bitcode_labels):
            aligned_matrix[task_idx, i] = dist.get(bitcode, 0.0)

    return aligned_matrix, bitcode_labels


def bitcode_distribution(latent_np_all, L):
    all_bitcodes = []
    # Iterate over all test sequences
    for index in range(latent_np_all.shape[0]):
        latent_np = latent_np_all[
            index, :, -L:
        ]  # Extract the last L components of the latent state for each sequence

        # Generate bitcodes based on the activity of the last L units for the current test sequence
        for row in latent_np:
            bitcode = "".join(["1" if value > 0 else "0" for value in row])
            all_bitcodes.append(bitcode)
    # Count the occurrences of each distinct bitcode across all test sequences
    bitcode_counter = Counter(all_bitcodes)
    # Number of distinct symbols (bitcodes)
    # num_distinct_symbols = len(bitcode_counter)
    # Distribution of bitcodes
    bitcode_distribution = {
        k: v / len(all_bitcodes) for k, v in bitcode_counter.items()
    }  # Normalize by total count

    return bitcode_distribution


def plot_bitcode_distribution(bitcode_distribution, threshold=0.001):
    # Filter out bitcodes with frequencies below the threshold
    filtered_bitcodes = {
        bitcode: freq
        for bitcode, freq in bitcode_distribution.items()
        if freq >= threshold
    }

    # Sort the bitcodes by their frequency in descending order
    sorted_bitcodes = sorted(
        filtered_bitcodes.items(), key=lambda item: item[1], reverse=True
    )

    # Check if there are any bitcodes left after filtering
    if len(sorted_bitcodes) == 0:
        print(f"No bitcodes with frequency above {threshold}.")
        return

    # Separate the bitcodes and their frequencies
    bitcodes, frequencies = zip(*sorted_bitcodes[:1000])  # Keep the top 50 if needed

    # Plot the sorted distribution
    plt.figure(figsize=(6, 4))
    plt.bar(bitcodes, frequencies, color="skyblue", edgecolor="black")

    # Add labels and title
    plt.xlabel("Bitcode")
    plt.ylabel("Frequency")
    # plt.xticks([])  # Remove x-axis tick labels completely

    # Show plot
    plt.tight_layout()
    plt.show()


def plot_bitcode_distributions(
    latent_states, model, task_names, aligned_matrix, bitcode_labels, top_k=20
):
    # Get top k most frequent bitcodes across all tasks
    total_freq = aligned_matrix.sum(axis=0)
    top_indices = np.argsort(total_freq)[-top_k:][::-1]

    aligned_matrix_top = aligned_matrix[:, top_indices]
    # bitcode_labels_top = [bitcode_labels[i] for i in top_indices]

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(aligned_matrix_top, aspect="auto", cmap="Blues")

    # Set ticks and labels
    ax.set_yticks(range(len(task_names)))
    ax.set_yticklabels(task_names)
    # ax.set_xticks(range(len(bitcode_labels_top)))
    # ax.set_xticks([])
    # ax.set_xticks(range(len(top_k)))
    # ax.set_xticklabels(bitcode_labels_top, rotation=45, fontsize=14)

    ax.set_xlabel("Top " + str(top_k) + " Bitcodes")
    ax.set_ylabel("Task")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Frequency", rotation=270, labelpad=15)

    plt.tight_layout()
    # plt.savefig('bitcode_distributions_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
