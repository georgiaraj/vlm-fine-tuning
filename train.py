import pdb
import os
import torch
import datetime
import argparse
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
        #print(f'Types of inputs: {inputs.keys()}')
        #print("Input shapes:", {k: v.shape if isinstance(v, torch.Tensor) else v for k, v in inputs.items()})
        return super().compute_loss(model, inputs, return_outputs=return_outputs)


def train(model, feature_extractor, tokenizer, train_dataset, val_dataset, args):

    def compute_metrics(eval_pred):
        pdb.set_trace()
        predictions, labels = eval_pred
        caption_preds = np.argmax(predictions[0], axis=-1)
        decoded_preds = tokenizer.batch_decode(caption_preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return rouge_score.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=3,
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

    return model, feature_extractor, tokenizer

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
    
    model, feature_extractor, tokenizer =  train(model, feature_extractor, tokenizer, train_dataset, val_dataset, args)

    print(f'Model trained')

    model.eval()

    print(f'Testing model on test dataset')
    with open('results.csv', 'w') as f:
        f.write('actual, predicted\n')
        for data in test_dataset:
            image = data['pixel_values'].unsqueeze(0).to(model.device)
            caption = tokenizer.decode(data['labels'], skip_special_tokens=True)
            outputs = model.generate(image)
            f.write(f'{caption},')
            f.write(f'{tokenizer.decode(outputs[0], skip_special_tokens=True)}\n')
        
    