import pdb
import os
import torch
import datetime
import argparse
import numpy as np
from pathlib import Path
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, VisionEncoderDecoderConfig
from transformers import Trainer, TrainingArguments
from evaluate import load

from custom_datasets import ScalpelDataset


rouge_score = load('rouge')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='nlpconnect/vit-gpt2-image-captioning')
    parser.add_argument('--dataset_path', type=str, default='scalpel-angles/scalpel_dataset.csv')
    parser.add_argument('--output_dir', type=str, default='scalpel-angles-model')
    parser.add_argument('--clearml-task-name', type=str, default=f'VLM Training Job {datetime.datetime.now()}')
    return parser.parse_args()


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs)

        return loss


def train(model, feature_extractor, tokenizer, train_dataset, val_dataset, args):

    def compute_metrics(eval_pred):
        #pdb.set_trace()
        predictions, labels = eval_pred
        caption_preds = np.argmax(predictions[0], axis=-1)
        decoded_preds = tokenizer.batch_decode(caption_preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        with open(f'val_results_{datetime.datetime.now()}.csv', 'w') as f:
            f.write('actual, pred\n')
            for label, pred in zip(decoded_labels, decoded_preds):
                f.write(f'{label},{pred}\n')
        return rouge_score.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        #warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        no_cuda=False,
        report_to=['clearml'],
        evaluation_strategy='epoch',
        save_strategy='epoch'
        )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    print(trainer.model.config)

    trainer.train()

    model.save_pretrained(args.output_dir)
    feature_extractor.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    return model, trainer

if __name__ == '__main__':

    args = parse_args()

    feature_extractor = ViTImageProcessor.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    data_dir = Path(args.dataset_path)

    os.environ.update({
        'CLEARML_LOG_MODEL': 'True',
        'CLEARML_TASK': args.clearml_task_name
    })

    dataset = ScalpelDataset(data_dir, feature_extractor, tokenizer)
    test_dataset = ScalpelDataset(data_dir, feature_extractor, tokenizer, data_set='test')
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    print(f'Train dataset length: {len(train_dataset)}')
    print(f'Validation dataset length: {len(val_dataset)}')
    print(f'Test dataset length: {len(test_dataset)}')

    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)
    
    print(model.config)
    #pdb.set_trace()

    #print(f'Model forward variables: {model.forward.__code__.co_varnames}')
    #print(f'Train dataset keys: {train_dataset[0].keys()}')

    #print(f'Model: {model}')
    #print(f'Feature extractor: {feature_extractor}')
    #print(f'Tokenizer: {tokenizer}')
    #pdb.set_trace()
    
    model, trainer =  train(model, feature_extractor, tokenizer, train_dataset, val_dataset, args)

    print(f'Model trained')

    model.eval()

    
    def test_model(model, dataset, output_file):
        predictions, label_ids, metrics = trainer.predict(dataset)
        with open(output_file, 'w') as f:
            f.write('actual, predicted\n')
            for data, pred in zip(dataset, predictions[0]):
                #pdb.set_trace()
                caption = tokenizer.decode(data['labels'], skip_special_tokens=True)
                p_caption = tokenizer.decode(np.argmax(pred, axis=-1), skip_special_tokens=True)
                f.write(f'{caption}, {pred}\n')
        

    with torch.no_grad():
    # Check result on validation dataset
        print(f'Testing model on val dataset')
        test_model(model, val_dataset, 'val_results.csv')

        print(f'Testing model on test dataset')
        test_model(model, test_dataset, 'test_results.csv')
        
    