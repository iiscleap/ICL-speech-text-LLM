import matplotlib.pyplot as plt
import numpy as np

def plot_results():
    # Data for HVB test dataset
    hvb_demonstrations = [0, 1, 2, 3, 4, 5, 10]
    hvb_data = {
        "No Warm-up (Q(S), D(T))": [1.26, 31.62, 22.22, 14.85, 14.56, 12.51, 15.98],
        "No Warm-up (Q(S), D(S))": [1.26, 50.07, 47.43, 42.91, 41.97, 42.04, 40.67],
        "HVB Matched (Q(S), D(T))": [37.89, 49.84, None, None, None, None, None],
        "HVB Matched (Q(S), D(S))": [37.89, None, None, None, None, None, None],
        "HVB Matched Symbol (Q(S), D(T))": [20.85, 34.59, 38.11, 40.03, 43.85, 45.58, None],
        "HVB Matched Symbol (Q(S), D(S))": [20.56, 50.20, 52.11, 50.65, 50.22, None, None],
        "Vox Mismatched (Q(S), D(T))": [0.83, 24.24, 22.71, 23.46, 21.67, 19.29, None],
        "Vox Mismatched (Q(S), D(S))": [0.85, 46.47, 40.23, 43.24, 42.87, 43.86, None],
        "Vox Mismatched Swap (Q(S), D(T))": [3.78, 41.05, 39.77, 42.19, 45.18, 46.97, 47.68],
        "Vox Mismatched Swap (Q(S), D(S))": [4.13, 51.16, 49.27, 48.18, 44.16, 42.44, 39.54]
    }

    # Data for Vox test dataset
    vox_demonstrations = [0, 1, 2, 3, 4, 5, 10]
    vox_data = {
        "No Warm-up (Q(S), D(T))": [36.16, 16.93, 20.71, 24.09, 25.04, 28.26, 23.20],
        "No Warm-up (Q(S), D(S))": [36.16, 37.06, 39.24, 35.47, 37.28, 38.49, 38.12],
        "HVB Mismatched (Q(S), D(T))": [35.57, 30.73, None, 37.48, 40.46, None, 40.53],
        "HVB Mismatched (Q(S), D(S))": [35.58, 41.48, 43.19, 38.19, 36.25, 32.65, 21.52],
        "HVB Mismatched Symbol (Q(S), D(T))": [42.84, 35.81, 41.29, 43.73, 46.06, 45.12, None],
        "HVB Mismatched Symbol (Q(S), D(S))": [42.87, 44.84, 48.37, 45.17, 43.63, 42.24, None],
        "Vox Matched (Q(S), D(T))": [61.06, 64.27, 63.28, 63.60, 64.28, 64.71, 62.71],
        "Vox Matched (Q(S), D(S))": [61.06, 62.45, 61.13, 58.12, 57.17, 57.26, None],
        "Vox Matched Swap (Q(S), D(T))": [46.43, 56.66, 54.23, 53.75, 53.72, 54.26, 54.27],
        "Vox Matched Swap (Q(S), D(S))": [46.43, 59.35, 60.68, 59.29, 57.25, 56.22, 57.80]
    }

    # Create two separate figures
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    for label, values in hvb_data.items():
        # Convert None to np.nan for plotting
        values = [np.nan if v is None else v for v in values]
        plt.plot(hvb_demonstrations, values, marker='o', label=label)

    plt.title('HVB Test Dataset Results')
    plt.xlabel('Number of Demonstrations')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.subplot(2, 1, 2)
    for label, values in vox_data.items():
        # Convert None to np.nan for plotting
        values = [np.nan if v is None else v for v in values]
        plt.plot(vox_demonstrations, values, marker='o', label=label)

    plt.title('Vox Test Dataset Results')
    plt.xlabel('Number of Demonstrations')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig('/data2/neeraja/neeraja/code/SALMONN/results/plots/demonstration_results.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    # Create results/plots directory if it doesn't exist
    import os
    os.makedirs('/data2/neeraja/neeraja/code/SALMONN/results/plots', exist_ok=True)
    plot_results()