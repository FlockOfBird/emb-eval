import json
import logging
import argparse

from transformers import AutoTokenizer, BertModel, BertForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_metric
import tqdm

import numpy as np

from imdb_data import get_imdb, get_small_imdb

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

# Read arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument("--Plot",
                    help="Draw plot for embeddings")
parser.add_argument("--Finetune_Pytorch",
                    help="Finetuning with full modification")
parser.add_argument("--Finetune_transformers",
                    help="Finetuning with Trainer")

args = parser.parse_args()

logging.info('loading dataset')
train_data, test_data = get_small_imdb(1000)

logging.info('loading tokenizer')
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def finetune_huggingface():
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2)

    def preprocess_function(examples):
        return bert_tokenizer(examples["text"], truncation=True)

    tokenized_train = train_data.map(preprocess_function, batched=True)
    tokenized_test = test_data.map(preprocess_function, batched=True)

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
    OUTPUT_PATH = './finetuned_bert'

    training_args = TrainingArguments(
        output_dir=OUTPUT_PATH,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=bert_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
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
scores = finetune_huggingface()

# Save the scores to a JSON file
with open('results/bert_finetune_results.json', 'w') as file:
    json.dump(scores, file)


def finetune_pytorch(epochs, dataloader, model, loss_fn, optimizer):
    logging.info('loading bert and tokenizer')

    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    logging.info('encoding data')
    # Prepare the text inputs for the model
    review_train = []
    for i, review in enumerate(train_data):
        tokens = bert_tokenizer.encode_plus(
            review['text'],
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = tokens['input_ids'].squeeze()
        token_type_ids = tokens['token_type_ids'].squeeze()
        mask = tokens["attention_mask"].squeeze()

        review_train.append({
            'input_ids': input_ids.clone().detach(),
            'token_type_ids': token_type_ids.clone().detach(),
            'mask': mask.clone().detach(),
            'target': torch.tensor(review['label'], dtype=torch.long)
        })

    dataloader = DataLoader(dataset=review_train, batch_size=32)

    logging.info('create model')

    class BERT(nn.Module):
        def __init__(self):
            super(BERT, self).__init__()
            self.bert_model = BertModel.from_pretrained("bert-base-uncased")
            self.out = nn.Linear(768, 1)

        def forward(self, ids, mask, token_type_ids):
            _, o2 = self.bert_model(ids,
                                    attention_mask=mask,
                                    token_type_ids=token_type_ids,
                                    return_dict=False
                                    )
            out = self.out(o2)
            return out

    model = BERT()

    loss_fn = nn.BCEWithLogitsLoss()

    # logging.info('freeze only last linear dense layer')
    # for param in model.bert_model.parameters():
    #     param.requires_grad = False

    # Initialize Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model.train()
    for epoch in range(epochs):
        print(epoch)

        # loop =
        for dl in dataloader:
            ids = dl['input_ids']
            token_type_ids = dl['token_type_ids']
            mask = dl['mask']
            label = dl['target']
            label = label.unsqueeze(1)

            optimizer.zero_grad()

            output = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids)
            label = label.type_as(output)

            loss = loss_fn(output, label)
            loss.backward()

            optimizer.step()

            pred = np.where(output >= 0, 1, 0)

            num_correct = sum(1 for a, b in zip(pred, label) if a[0] == b[0])
            num_samples = pred.shape[0]
            accuracy = num_correct/num_samples

            print(
                f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

            # Show progress while training
            print(f'Epoch={epoch}/{epochs}')
            loss_val = loss.item()
            print(loss_val, accuracy)

    return model
