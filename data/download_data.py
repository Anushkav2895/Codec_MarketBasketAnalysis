import kagglehub, shutil, os

path = kagglehub.dataset_download("heeraldedhia/groceries-dataset")
print("Path to dataset files:", path)

os.makedirs("data", exist_ok=True)
for f in os.listdir(path):
    if f.endswith(".csv"):
        shutil.copy(os.path.join(path, f), os.path.join("data", f))
        print("Copied:", f)