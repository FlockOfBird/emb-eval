import logging
import json
import argparse

import torch
from transformers import LlamaTokenizer, LlamaForSequenceClassification, TrainingArguments, Trainer,TFTrainer, DataCollatorWithPadding
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import AdamW

from datasets import load_metric

from accelerate import infer_auto_device_map

from imdb_data import get_imdb, get_small_imdb

# logging configuration for better code monitoring
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

# Read arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument("--Plot", help="Draw plot for embeddings")
args = parser.parse_args()


PATH_TO_CONVERTED_WEIGHTS = "./7B_converted/"

logging.info('loading dataset')
train_data, test_data = get_small_imdb(500)

# Use GPU runtime an High_RAM before run this pieace of code
print('if cuda is available:', torch.cuda.is_available())  # Default CUDA device
print('current cuda device:', torch.cuda.current_device())  # returns 0 in my case
print('number of cuda devices', torch.cuda.device_count()) 

logging.info('loading model and tokenizer')

llama_model = LlamaForSequenceClassification.from_pretrained(
    PATH_TO_CONVERTED_WEIGHTS,
    device_map="auto",
    num_labels=2
)

llama_tokenizer = LlamaTokenizer.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)

llama_tokenizer.pad_token_id = (
    0  # unknow tokens. we want this to be different from the eos token
)
llama_tokenizer.padding_side = "left"

# def finetune_pytorch(model):
logging.info('tokenizing and loading data into gpus')
def preprocess_function(examples):
    return llama_tokenizer(examples["text"], truncation=True)

tokenized_train = train_data.map(preprocess_function, batched=True)
tokenized_test = test_data.map(preprocess_function, batched=True)

tokenized_train = DataLoader(
    tokenized_train,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
tokenized_test = DataLoader(
    tokenized_test,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

optimizer = AdamW(llama_model.parameters(), lr=3e-4)
llama_model = nn.DataParallel(llama_model, device_ids=[0, 1, 2, 3])

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(device)
llama_model.to(device)



def finetune_huggingface():
    def preprocess_function(examples):
        return llama_tokenizer(examples["text"], truncation=True)

    tokenized_train = train_data.map(preprocess_function, batched=True)
    tokenized_test = test_data.map(preprocess_function, batched=True)

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
    OUTPUT_PATH = './finetuned_bert'
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
    print('device:',training_args.device)

    trainer = Trainer(
        model=llama_model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=llama_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # trainer.args._n_gpu = 4
    # print(trainer)
    print(trainer.args)
    # print(dir(trainer))

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
# scores = finetune_huggingface()

# Save the scores to a JSON file
with open('results/llama_finetune_results.json', 'w') as file:
    json.dump(scores, file)

# prompt = "Hey, are you conscious? Can you talk to me?"
# inputs = tokenizer(prompt, return_tensors="pt")

# # simple code to generate code and check the model funtionality
# generate_ids = model.generate(inputs.input_ids, max_length=30)
# tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
#                        clean_up_tokenization_spaces=False)[0]


# def get_embeddings(model, tokenizer, prompt):
#     input_ids = tokenizer(prompt).input_ids
#     # we can use get_output_embeddings as well
#     input_embeddings = model.get_input_embeddings()
#     embeddings = input_embeddings(torch.LongTensor([input_ids]))
#     mean = torch.mean(embeddings[0], 0).cpu().detach()
#     return mean


# prompt1 = "That is a happy dog"
# prompt2 = "That is a very happy person"
# prompt3 = "Today is a sunny day"

# emb1 = get_embeddings(model, tokenizer, prompt1)
# emb2 = get_embeddings(model, tokenizer, prompt2)
# emb3 = get_embeddings(model, tokenizer, prompt3)


# def get_cosine_similarity(feature_vec_1, feature_vec_2):
#     return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]


# print(get_cosine_similarity(emb1, emb2))
# print(get_cosine_similarity(emb2, emb3))
# print(get_cosine_similarity(emb1, emb3))
