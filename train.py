import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import Trainer, TrainingArguments

from datasets import ScalpelDataset


def train(train_dataset, val_dataset):
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    model.save_pretrained('scalpel-angles-model')
    feature_extractor.save_pretrained('scalpel-angles-feature-extractor')
    tokenizer.save_pretrained('scalpel-angles-tokenizer')

    return model, feature_extractor, tokenizer

if __name__ == '__main__':
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    dataset, test_dataset = ScalpelDataset('../scalpel-angles/scalpel_dataset.csv', feature_extractor, tokenizer)
    train_dataset, val_dataset = dataset.train_val_split(0.9)

    model, feature_extractor, tokenizer =  train(model, feature_extractor, tokenizer, train_dataset, val_dataset)

    model.eval()

    for image, caption in test_dataset:
        inputs = feature_extractor(image, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=20)
        print(f'predicted: {tokenizer.decode(outputs[0], skip_special_tokens=True)}')
        print(f'actual: {caption}')
        print('-'*100)