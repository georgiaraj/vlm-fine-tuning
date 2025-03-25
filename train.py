import pdb
import torch
import argparse
from pathlib import Path
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import Trainer, TrainingArguments
from evaluate import load

from custom_datasets import ScalpelDataset


rouge_score = load('rouge')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='nlpconnect/vit-gpt2-image-captioning')
    parser.add_argument('--dataset_path', type=str, default='scalpel-angles/scalpel_dataset.csv')
    parser.add_argument('--output_dir', type=str, default='scalpel-angles-model')
    return parser.parse_args()


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        print(f'Types of inputs: {inputs.keys()}')
        print("Input shapes:", {k: v.shape if isinstance(v, torch.Tensor) else v for k, v in inputs.items()})
        return super().compute_loss(model, inputs, return_outputs=return_outputs)


def train(model, feature_extractor, tokenizer, train_dataset, val_dataset, args):

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return rouge_score.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        no_cuda=False,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

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

    dataset = ScalpelDataset(data_dir, feature_extractor, tokenizer)
    test_dataset = ScalpelDataset(data_dir, feature_extractor, tokenizer, data_set='test')
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    print(f'Train dataset length: {len(train_dataset)}')
    print(f'Validation dataset length: {len(val_dataset)}')
    print(f'Test dataset length: {len(test_dataset)}')

    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)

    print(f'Model forward variables: {model.forward.__code__.co_varnames}')
    print(f'Train dataset keys: {train_dataset[0].keys()}')

    #print(f'Model: {model}')
    #print(f'Feature extractor: {feature_extractor}')
    #print(f'Tokenizer: {tokenizer}')

    model, feature_extractor, tokenizer =  train(model, feature_extractor, tokenizer, train_dataset, val_dataset, args)

    print(f'Model trained')

    model.eval()

    print(f'Testing model on test dataset')
    for image, caption in test_dataset:
        inputs = feature_extractor(image, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=20)
        print(f'predicted: {tokenizer.decode(outputs[0], skip_special_tokens=True)}')
        print(f'actual: {caption}')
        print('-'*100)