import json
import logging

import evaluate

from transformers import AutoTokenizer, BertModel, BertForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, get_scheduler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import load_metric
from tqdm.auto import tqdm

import numpy as np
from data import *

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

logging.info('loading dataset')
db = get_small_dataset("imdb")
train_data, test_data = db["train"], db["train"]

logging.info('loading model')
model = BertForSequenceClassification.from_pretrained(
    "bert-base-cased", num_labels=2)
# model = torch.nn.DataParallel(model, device_ids=[0])

logging.info('loading tokenizer')
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

logging.info('tokenizing and loading data into gpus')


def preprocess_function(examples):
    return bert_tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_train = train_data.map(preprocess_function, batched=True)
tokenized_test = test_data.map(preprocess_function, batched=True)

optimizer = AdamW(model.parameters(), lr=3e-4)


def finetune_pytorch(tokenized_train, tokenized_test, optimizer):
    # remove unrelated columns
    tokenized_train = tokenized_train.remove_columns(["text", "Unnamed: 0"])
    tokenized_test = tokenized_test.remove_columns(["text",  "Unnamed: 0"])
    # rename label
    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_test = tokenized_test.rename_column("label", "labels")

    tokenized_train.set_format("torch")
    tokenized_test.set_format("torch")

    train_dataloader = DataLoader(tokenized_train, shuffle=True, batch_size=8)
    test_dataloader = DataLoader(tokenized_test, batch_size=8)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    progress_bar = tqdm(range(num_training_steps))

    print('change the model mode to train')
    model.train()

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            # batch = {k: v.to('cuda:0') for k, v in batch.items()}
            # input = {
            #     'labels': batch['labels'].to('cuda:0'),
            #     'input_ids': batch['input_ids'].to('cuda:0'),
            #     'attention_mask': batch['attention_mask'].to('cuda:0')
            # }
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    metric = evaluate.load("accuracy")

    model.eval()
    for batch in test_dataloader:
        batch = {k: v.to('cuda:0') for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    metric.compute()


def finetune_huggingface(tokenized_train, tokenized_test, optimizer):

    data_collator = DataCollatorWithPadding(tokenizer=bert_tokenizer)
    # Define the evaluation metrics

    def compute_metrics(eval_pred):
        load_accuracy = load_metric("accuracy")
        load_f1 = load_metric("f1")

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = load_accuracy.compute(
            predictions=predictions, references=labels)["accuracy"]
        f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
        return {"accuracy": accuracy, "f1": f1}

    # Define a new Trainer with all the objects we constructed so far
    OUTPUT_PATH = './finetuned_llama'

    training_args = TrainingArguments(
        output_dir=OUTPUT_PATH,
        learning_rate=3e-4,
        # per_gpu_train_batch_size=16,
        # per_gpu_eval_batch_size=16,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        strategy='ddp_sharded',
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=bert_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Compute the evaluation metrics
    result = trainer.evaluate()

    # experiment descriptions
    scores = {
        'Data': 'IMDB',
        'Data Size Test': len(tokenized_train),
        'Data Size Train': len(tokenized_test),
        'F1 score': result["eval_f1"],
        'Accuracy': result["eval_accuracy"]
    }
    return scores


logging.info('finetuning model')
# scores = finetune_huggingface(tokenized_train, tokenized_test, optimizer)
scores = finetune_pytorch(tokenized_train, tokenized_test, optimizer)

# Save the scores to a JSON file
with open('results/bert_finetune_results.json', 'w') as file:
    json.dump(scores, file)
