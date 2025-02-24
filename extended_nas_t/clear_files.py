# List of file paths to clear
file_paths = [
    "/abyss/home/NAS-T/nas_t/accuracies.json",
    "/abyss/home/NAS-T/nas_t/model_sizes.json",
    "/abyss/home/NAS-T/nas_t/evolutionary_runs.json",
    "/abyss/home/NAS-T/nas_t/standard_results.json"
]

# Iterate over the file paths and clear each file
for file_path in file_paths:
    try:
        with open(file_path, 'w') as file:
            file.truncate(0)  # Clear the content of the file
        print(f"Cleared content of: {file_path}")
    except Exception as e:
        print(f"Failed to clear {file_path}: {e}")
