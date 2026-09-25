from collections import Counter
import matplotlib.pyplot as plt



def bitcode_distribution(latent_states, L):

    latent_np_all = latent_states.detach().numpy()

    # Initialize a list to store all bitcodes across all test sequences
    all_bitcodes = []

    # Iterate over all test sequences
    for index in range(latent_np_all.shape[0]):
        latent_np = latent_np_all[index, :, -L:]  # Extract the last L components of the latent state for each sequence

        # Generate bitcodes based on the activity of the last L units for the current test sequence
        for row in latent_np:
            bitcode = ''.join(['1' if value > 0 else '0' for value in row])
            all_bitcodes.append(bitcode)

    # Count the occurrences of each distinct bitcode across all test sequences
    bitcode_counter = Counter(all_bitcodes)

    # Number of distinct symbols (bitcodes)
    num_distinct_symbols = len(bitcode_counter)

    # Distribution of bitcodes
    bitcode_distribution = {k: v / len(all_bitcodes) for k, v in bitcode_counter.items()}  # Normalize by total count
    
    return bitcode_distribution

def plot_bitcode_distribution(bitcode_distribution, threshold=0.001):
    # Filter out bitcodes with frequencies below the threshold
    filtered_bitcodes = {bitcode: freq for bitcode, freq in bitcode_distribution.items() if freq >= threshold}
    
    # Sort the bitcodes by their frequency in descending order
    sorted_bitcodes = sorted(filtered_bitcodes.items(), key=lambda item: item[1], reverse=True)
    
    # Check if there are any bitcodes left after filtering
    if len(sorted_bitcodes) == 0:
        print(f"No bitcodes with frequency above {threshold}.")
        return
    
    # Separate the bitcodes and their frequencies
    bitcodes, frequencies = zip(*sorted_bitcodes[:1000])  # Keep the top 50 if needed

    # Plot the sorted distribution
    plt.figure(figsize=(6, 4))
    plt.bar(bitcodes, frequencies, color='skyblue', edgecolor='black')
    
    # Add labels and title
    plt.xlabel('Bitcode')
    plt.ylabel('Frequency')
    plt.xticks([])  # Remove x-axis tick labels completely
    
    # Show plot
    plt.tight_layout()
    plt.show()