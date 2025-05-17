"""
Дообучение RuT5 на генерацию заголовков с расширенным функционалом:
- аргументы командной строки
- логирование
- DataCollator
- cosine scheduler с hard restarts
- early stopping
- вычисление метрик ROUGE и BLEU
- mixed precision
"""
import os
import argparse
import logging
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    get_cosine_with_hard_restarts_schedule_with_warmup
)
from datasets import Dataset, load_metric


def parse_args():
    parser = argparse.ArgumentParser(description="Train RuT5 for headline generation")
    parser.add_argument('--model_name', type=str, default='cointegrated/ruT5-base', help='Pretrained model')
    parser.add_argument('--data_csv', type=str, required=True, help='Path to Lenta.ru CSV')
    parser.add_argument('--output_dir', type=str, default='./headline_model', help='Where to save')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_input', type=int, default=512)
    parser.add_argument('--max_target', type=int, default=51)
    return parser.parse_args()


def compute_metrics(eval_pred):
    metric_rouge = load_metric('rouge')
    metric_bleu = load_metric('bleu')
    preds, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Rouge expects a newline after each sentence
    result_rouge = metric_rouge.compute(predictions=decoded_preds, references=decoded_labels)
    bleu_preds = [pred.split() for pred in decoded_preds]
    bleu_refs = [[ref.split()] for ref in decoded_labels]
    result_bleu = metric_bleu.compute(predictions=bleu_preds, references=bleu_refs)
    return {
        'rouge-l': result_rouge['rougeL'].mid.fmeasure,
        'bleu': result_bleu['bleu']
    }

class HeadlineTrainer:
    def __init__(self, args, device: torch.device):
        self.args = args
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)

    def load_dataset(self):
        import pandas as pd
        df = pd.read_csv(self.args.data_csv)
        df = df[['text', 'headline']]
        ds = Dataset.from_pandas(df)
        return ds.train_test_split(test_size=0.15, seed=42)

    def preprocess(self, examples):
        inputs = self.tokenizer(examples['text'], truncation=True, padding='max_length', max_length=self.args.max_input)
        targets = self.tokenizer(examples['headline'], truncation=True, padding='max_length', max_length=self.args.max_target)
        inputs['labels'] = targets['input_ids']
        return inputs

    def train(self):
        # загрузка и препроцессинг
        splits = self.load_dataset()
        train_ds = splits['train'].map(self.preprocess, batched=True, remove_columns=['text', 'headline'])
        val_ds = splits['test'].map(self.preprocess, batched=True, remove_columns=['text', 'headline'])

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.args.output_dir,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            num_train_epochs=self.args.epochs,
            learning_rate=self.args.lr,
            weight_decay=0.01,
            logging_dir=os.path.join(self.args.output_dir, 'logs'),
            logging_steps=100,
            save_total_limit=3,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=2,
            lr_scheduler_type='cosine_with_restarts',
            warmup_steps=500
        )
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks
        )
        trainer.train()
        trainer.save_model(self.args.output_dir)

if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(level=logging.INFO)
    trainer = HeadlineTrainer(args, device)
    trainer.train()
