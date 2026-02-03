from jaxflow.core.dataset import Dataset
from jaxflow.core.loader import Loader

class MnistDataset(Dataset):
    def __len__(self):
        return 100
    
    def __getitem__(self, idx):
        return {"image": [0.0]*784, "label": 0}

def main():
    ds = MnistDataset()
    loader = Loader(ds)
    print(f"Dataset length: {len(ds)}")
    # Simulate training loop
    for batch in loader:
        print(batch)

if __name__ == "__main__":
    main()
