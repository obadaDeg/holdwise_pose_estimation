import re
import pandas as pd

def extract_training_data(file_path, output_csv1, output_csv2):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Splitting the different training approaches
    sections = content.split("this should be speerated becuase its from another approach")
    
    # Improved regex pattern to capture variations in log formatting
    pattern = re.compile(
        r"Epoch\s+(\d+)/(\d+).*?- loss:\s*([\d\.]+)\s*- accuracy:\s*([\d\.]+)\s*- val_loss:\s*([\d\.]+)\s*- val_accuracy:\s*([\d\.]+)"
    )

    dataframes = []
    for section in sections:
        matches = pattern.findall(section)
        if matches:
            df = pd.DataFrame(matches, columns=["Epoch", "Total Epochs", "Loss", "Accuracy", "Val Loss", "Val Accuracy"])
            df = df.astype({"Epoch": int, "Total Epochs": int, "Loss": float, "Accuracy": float, "Val Loss": float, "Val Accuracy": float})
            dataframes.append(df)
        else:
            print("No matches found in a section. Check formatting.")
    
    # Save the extracted data to CSV files
    if len(dataframes) > 0:
        dataframes[0].to_csv(output_csv1, index=False)
    if len(dataframes) > 1:
        dataframes[1].to_csv(output_csv2, index=False)
    
    return dataframes

# Example usage
file_path = "text.txt"  # Replace with your file path
output_csv1 = "training_approach_1.csv"
output_csv2 = "training_approach_2.csv"
dataframes = extract_training_data(file_path, output_csv1, output_csv2)

# Display extracted data
for i, df in enumerate(dataframes):
    csv_filename = f"training_approach_{i+1}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}")
    print(df.head())
