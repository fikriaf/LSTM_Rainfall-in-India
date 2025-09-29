import kagglehub

# Download latest version
path = kagglehub.dataset_download("rajanand/rainfall-in-india")

print("Path to dataset files:", path)