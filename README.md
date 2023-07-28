## Always monitor the GPUs while using delta, matrix
you can monitor the system using these commands:

```bash
$ nvidia-smi
$ glances
$ htop
```
check GPU status
```python
import torch

print('if cuda is available:', torch.cuda.is_available())
print('current cuda device:', torch.cuda.current_device())
print('number of cuda devices', torch.cuda.device_count())
```
## Prepare enviornment
Before running any python file, first active virtual environment with 
```bash
$ source env/bin/activate
```
You can also install requirements if want to run the code on a different server. First freeze the requirements and then copy the requirements.txt file to your destination server and run `pip install -r requirements.txt`. Note that converted weights must be reconverted if you are changing the directories or servers. NERVER change the path of *B_converted directories on this server.

## Flags (Arguments)
You can plot embeddings if you pass `--Plot True` argument. for example: 
```bash
$ python bert_base.py --Plot True
```
___
## Deal with huge models
### See layers
This is an example of how you can see the layers of your model:
```python
device = {"block1": 0, "block2": 1, "block3": 2, "block4": 3 }

config = LlamaConfig.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
with init_empty_weights():
    self.model = LlamaForCausalLM._from_config(config)

device = infer_auto_device_map(self.model)
print(device)
```

### LLaMA 30B
To run llama-30b you need to change the code for loading model to this:
```python
device = "auto"
self.model = LlamaForCausalLM.from_pretrained(
    PATH_TO_CONVERTED_WEIGHTS,
    device_map=device,
    max_memory={0: "12GiB", 1: "12GiB", 2:"12GiB", 3:"12GiB"},
    offload_folder="offload"
)
```
Since a lot of layers are going to load in cpu in this approach, the loading time of the model is very high (nearly 45 minutes).
### Other LLaMAs
```python
device = "auto"
self.model = LlamaForCausalLM.from_pretrained(
    PATH_TO_CONVERTED_WEIGHTS,
    device_map=device
)
```