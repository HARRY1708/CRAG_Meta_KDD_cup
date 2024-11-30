import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the JSON file
file_path = '/home/jupyter/cs245-project-crag-master/output/classified_queries.json'

# Output folder and file path for the heatmap
output_folder = '/home/jupyter/cs245-project-crag-master/output/query_classifier_data'
heatmap_output_file = os.path.join(output_folder, 'ground_truth_classification_percentage_heatmap.png')

# Load the JSON data
try:
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Extract ground truth and classification data
    ground_truth_types = [item['ground_truth_type'] for item in data]
    classifications = [item['classification'] for item in data]

    # Create a DataFrame to represent the confusion matrix
    unique_types = sorted(set(ground_truth_types + classifications))
    matrix = pd.DataFrame(0, index=unique_types, columns=unique_types)

    # Populate the matrix
    for gt, cl in zip(ground_truth_types, classifications):
        matrix.loc[gt, cl] += 1

    # Calculate percentage matrix for the heatmap
    percentage_matrix = matrix.div(matrix.sum(axis=1), axis=0) * 100

    # Print the confusion matrix (number of examples) in the terminal
    print("\nGround Truth vs Classification Matrix (Rows: Ground Truth, Columns: Classification):")
    print(matrix)

    # Generate the percentage heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(percentage_matrix, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)
    plt.title('Ground Truth vs Classification Percentage Heatmap')
    plt.xlabel('Classified As')
    plt.ylabel('Ground Truth Type')

    # Save the percentage heatmap
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(heatmap_output_file)
    plt.show()
    print(f"\nPercentage heatmap saved to: {heatmap_output_file}")

    # Plot total examples per class as a bar chart
    total_examples_per_class = matrix.sum(axis=1)
    plt.figure(figsize=(10, 6))
    total_examples_per_class.sort_values(ascending=False).plot(kind='bar', color='skyblue')
    plt.title('Total Examples Per Ground Truth Class')
    plt.xlabel('Ground Truth Type')
    plt.ylabel('Total Examples')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the bar chart
    bar_chart_output_file = os.path.join(output_folder, 'total_examples_per_class.png')
    plt.savefig(bar_chart_output_file)
    plt.show()
    print(f"\nBar chart saved to: {bar_chart_output_file}")

except FileNotFoundError:
    print(f"File not found: {file_path}. Please check the path.")
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
except KeyError as e:
    print(f"Missing expected key in data: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
