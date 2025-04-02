import pdb
import os
import torch
import datetime
import argparse
import numpy as np
from pathlib import Path
from evaluate import load

from custom_datasets import ScalpelDataset
from model import models, VLMModel


rouge_score = load('rouge')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default='vision-encoder-decoder', choices=models.keys())
    parser.add_argument('--model-name', type=str, default='nlpconnect/vit-gpt2-image-captioning')
    parser.add_argument('--dataset-path', type=str, default='scalpel-angles/scalpel_dataset.csv')
    parser.add_argument('--output-dir', type=str, default='scalpel-angles-model')
    parser.add_argument('--clearml-task-name', type=str, default=f'VLM Training Job {datetime.datetime.now()}')
    parser.add_argument('--num-epochs', type=int, default=100, help="Number of epochs to train")
    parser.add_argument('--batch-size', type=int, default=8, help="Batch size for training")
    parser.add_argument('--learning-rate', type=float, default=5e-5, help="Learning rate for training")
    parser.add_argument('--weight-decay', type=float, default=0.01, help="Weight decay for training")
    parser.add_argument('--verbose', action='store_true', help="Verbose logging")
    return parser.parse_args()



if __name__ == '__main__':

    args = parse_args()

    model = VLMModel(args.model_name, args.model_type, args)

    data_dir = Path(args.dataset_path)

    os.environ.update({
        'CLEARML_LOG_MODEL': 'True',
        'CLEARML_TASK': args.clearml_task_name
    })

    dataset = ScalpelDataset(data_dir, model.feature_extractor, model.tokenizer)
    test_dataset = ScalpelDataset(data_dir, model.feature_extractor, model.tokenizer,
                                  data_set='test')
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    print(f'Train dataset length: {len(train_dataset)}')
    print(f'Validation dataset length: {len(val_dataset)}')
    print(f'Test dataset length: {len(test_dataset)}')

    print(model.config)
    #pdb.set_trace()

    #print(f'Model forward variables: {model.forward.__code__.co_varnames}')
    #print(f'Train dataset keys: {train_dataset[0].keys()}')

    #print(f'Model: {model}')
    #print(f'Feature extractor: {feature_extractor}')
    #print(f'Tokenizer: {tokenizer}')
    #pdb.set_trace()

    model.train(train_dataset, val_dataset, args)

    print(f'Model trained')

    print(f'Testing model on val dataset')
    model.evaluate(val_dataset)

    print(f'Testing model on test dataset')
    model.evaluate(test_dataset)
