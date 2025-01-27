from PIL import Image
from torch.utils.data import Dataset
import pandas as pd


class ScalpelDataset(Dataset):
    def __init__(self, csv_file, feature_extractor, tokenizer):
        self.data = pd.read_csv(csv_file, columns=['image', 'caption'])
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 'image']
        image = Image.open(image_path).convert('RGB')
        features = self.feature_extractor(image, return_tensors="pt")
        caption = self.data.iloc[idx, 'caption']
        tokenized_caption = self.tokenizer(caption, padding="max_length", truncation=True, max_length=128)
        return features, tokenized_caption