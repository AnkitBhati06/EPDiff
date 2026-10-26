import os
import shutil

import kagglehub

project_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(project_dir, "data")
cache_dir = os.path.join(data_dir, ".kagglehub")
os.makedirs(cache_dir, exist_ok=True)
os.environ["KAGGLEHUB_CACHE"] = cache_dir

# Download dataset
path = kagglehub.dataset_download("dschettler8845/brats-2021-task1")
print("Downloaded at:", path)

# Create target directory beside this script, regardless of the current directory.
target = os.path.join(data_dir, "BraTS21")
os.makedirs(target, exist_ok=True)

# Copy files into project
shutil.copytree(path, target, dirs_exist_ok=True)

print("Dataset copied to:", target)
