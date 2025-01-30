import pandas as pd
from PIL import Image
import pdb
from torch.utils.data import Dataset


class ScalpelDataset(Dataset):
    def __init__(self, data_dir, feature_extractor, tokenizer, data_set='train'):
        if data_set not in ['train', 'test']:
            raise ValueError(f'Invalid data set: {data_set}')
        self.data = pd.read_csv(data_dir / 'scalpel_dataset.csv', usecols=['image', 'caption', 'data_set'])
        self.data = self.data[self.data['data_set'] == data_set]
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.root_dir = data_dir
        self.data_set = data_set

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 0]
        image = Image.open(self.root_dir / 'images' / image_path).convert('RGB')
        features = self.feature_extractor(image, return_tensors="pt")
        caption = self.data.iloc[idx, 1]
        tokenized_caption = self.tokenizer(caption, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        return {
            'pixel_values': features['pixel_values'], 
            'input_ids': tokenized_caption['input_ids'], 
            'attention_mask': tokenized_caption['attention_mask']
        }
