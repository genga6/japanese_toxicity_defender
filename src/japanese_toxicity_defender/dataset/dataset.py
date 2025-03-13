class ToxicDataset(Dataset):
    def __init__(self, df):
        self.tokens = df["tokens"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.texts[idx]), 
            "labels": torch.tensor(self.labels[idx])
        }

