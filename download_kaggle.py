import kagglehub
try:
    path = kagglehub.dataset_download("naureenmohammad/mmu-iris-dataset")
    print("Path to dataset files:", path)
except Exception as e:
    print("Error:", e)
