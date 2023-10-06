## GPU
make sure you have nvidia installed on your os before you run the code.
```bash
$ nvidia-smi
```
You can also check the GPU status code using following pytorch interface.
```python
import torch

print('if cuda is available:', torch.cuda.is_available())
print('current cuda device:', torch.cuda.current_device())
print('number of cuda devices', torch.cuda.device_count())
```

## Setup
Before running anything, make sure to install the required dependencies in a virtual environment. You can install these dependencies using the following command:
```bash
pip install -r requirements.txt
```

To load LLaMA you need to convert original LLaMA weights to pytorch format.
```bash
python convert_llama_weights_to_hf.py \
    --input_dir /path/to/original/llama/weights --model_size 7B --output_dir /output/path
```
After conversion, in the get_llama_embedding.py replace models_path with your --output_dir.

## Get Embeddings
To get embeddings you need to run `main.py`. Select the datasets and models from which you want to obtain embeddings in the `main.py` and run:
```bash
python main.py
```
