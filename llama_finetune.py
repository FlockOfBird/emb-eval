import logging
import json
from tqdm.auto import tqdm

import evaluate

from transformers import LlamaTokenizer, LlamaForSequenceClassification, TrainingArguments, Trainer, TFTrainer, DataCollatorWithPadding, get_scheduler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.distributed as dist
from datasets import load_metric

from accelerate import infer_auto_device_map

from data import *

# logging configuration for better code monitoring
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

PATH_TO_CONVERTED_WEIGHTS = "./llama_converted/7B/"

logging.info('loading dataset')
db = get_small_dataset("imdb")
train_data, test_data = db["train"], db["train"]

# Use GPU runtime an High_RAM before run this pieace of code
print('if cuda is available:', torch.cuda.is_available())  # Default CUDA device
print('current cuda device:', torch.cuda.current_device())  # returns 0 in my case
print('number of cuda devices', torch.cuda.device_count())

logging.info('loading model and tokenizer')

device = "auto"  # balanced_low_0, auto, balanced, sequential

# model = LlamaForSequenceClassification.from_pretrained(
#     PATH_TO_CONVERTED_WEIGHTS,
#     device_map=device,
#     max_memory={0: "0GiB", 1: "0GiB", 2: "15GiB", 3: "15GiB"},
#     offload_folder="offload",
#     num_labels=2
# )

model = LlamaForSequenceClassification.from_pretrained(
    PATH_TO_CONVERTED_WEIGHTS,
    device_map=device,
    num_labels=2
)
# model.cuda()


def list_model_layers(model):
    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())
    print('The LLaMA model has {:} different named parameters.\n'.format(
        len(params)))
    for p in params:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


llama_tokenizer = LlamaTokenizer.from_pretrained(
    PATH_TO_CONVERTED_WEIGHTS, model_max_length=512)
llama_tokenizer.pad_token_id = (
    0  # unknow tokens. we want this to be different from the eos token
)
llama_tokenizer.padding_side = "left"

logging.info('tokenizing and loading data into gpus')


def preprocess_function(examples):
    return llama_tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_train = train_data.map(preprocess_function, batched=True)
tokenized_test = test_data.map(preprocess_function, batched=True)

optimizer = AdamW(model.parameters(), lr=3e-4)


def reset_everything():
    del model
    torch.cuda.empty_cache()


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
    print('bulding scheduler')
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    progress_bar = tqdm(range(num_training_steps))
    print('change the model mode to train')
    model.train()

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to('cuda:0') for k, v in batch.items()}
            input = {
                'labels': batch['labels'].to('cuda'),
                'input_ids': batch['input_ids'].to('cuda'),
                'attention_mask': batch['attention_mask'].to('cuda')
            }
            outputs = model(**input)
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

    data_collator = DataCollatorWithPadding(tokenizer=llama_tokenizer)
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
        tokenizer=llama_tokenizer,
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
with open('results/llama_finetune_results.json', 'w') as file:
    json.dump(scores, file)
