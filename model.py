import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, VisionEncoderDecoderConfig
from transformers import Trainer, TrainingArguments


models = {
    'vision-encoder-decoder': (VisionEncoderDecoderModel, ViTImageProcessor),
}


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs)
        # Additional custom loss computation here if needed
        return loss


class VLMModel():

    def __init__(self, model_name, model_type, args):
        self.model_name = model_name
        self.model_type, self.image_processor = models[model_type]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.feature_extractor = self.image_processor.from_pretrained(model_name)
        self.model = self.model_type.from_pretrained(model_name)

        self.config = self.model_type.from_pretrained(model_name)
        self.config.decoder_start_token_id = self.tokenizer.pad_token_id
        self.config.eos_token_id = self.tokenizer.eos_token_id
        self.config.pad_token_id = self.tokenizer.pad_token_id

        self.verbose = args.verbose
        self.output_dir = args.output_dir

        self.trainer = None
        self.training_args = None

    def _decode_preds_labels(self, predictions, labels):
        caption_preds = np.argmax(predictions, axis=-1)
        decoded_preds = self.tokenizer.batch_decode(caption_preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        return decoded_preds, decoded_labels

    def _compute_metrics(self, eval_pred):
        decoded_preds, decoded_labels = self._decode_preds_labels(*eval_pred)
        if self.verbose:
            self._write_result(decoded_labels, decoded_preds, prefix='val')
        return rouge_score.compute(predictions=decoded_preds,
                                   references=decoded_labels, use_stemmer=True)

    def _write_result(self, labels, preds, prefix='evaluation'):
        with open(f'{prefix}_results_{datetime.datetime.now()}.csv', 'w') as f:
            f.write('actual, pred\n')
            for label, pred in zip(decoded_labels, decoded_preds):
                f.write(f'{label},{pred}\n')

    def _trainer(self, training_args, train_dataset, val_dataset):
        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
        )

    def train(self, train_dataset, val_dataset, args):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.num_epochs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir='./logs',
            logging_steps=10,
            load_best_model_at_end=True,
            no_cuda=False,
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            report_to="clearml",
        )

        self.trainer = self._trainer(training_args, train_dataset, val_dataset)

        self.trainer.train()

    def evaluate(self, eval_dataset, prefix='val'):
        self.model.eval()
        predictions, label_ids, metrics = self.trainer.predict(eval_dataset)

        print(f"Evaluation metrics for {prefix} set: {metrics}")

        if self.verbose:
            decoded_preds, decoded_labels = self._decode_preds_labels(predictions, label_ids)
            self._write_result(decoded_labels, decoded_preds, prefix=prefix)

    def save(self):
        self.model.save_pretrained(self.output_dir)
        self.feature_extractor.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

    def load(self):
        self.model = self.model_type.from_pretrained(self.output_dir)
        self.feature_extractor = self.image_processor.from_pretrained(self.output_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.output_dir)

    def print(self):
        print(f'Model name: {self.model_name}')
        print(f'Model type: {self.model_type}')
        print(f'Feature extractor: {self.feature_extractor}')
        print(f'Tokenizer: {self.tokenizer}')
        print(f'Config: {self.config}')

        print(f'Model: {self.model}')
        print(f'Model forward variables: {self.model.forward.__code__.co_varnames}')
