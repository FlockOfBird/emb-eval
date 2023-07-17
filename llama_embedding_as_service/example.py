from embedbase_client import EmbedbaseClient

client = EmbedbaseClient("http://localhost:8000")
SENTENCES = [
    "LLaMA is a large language model.",
    "LLaMA is an animal living in the mountains.",
    "Alpaca is instruction based version of LLaMA.",
]
DATASET_ID = "LLaMAs"
data = client.dataset(DATASET_ID).batch_add([{"data": sentence} for sentence in SENTENCES])
print(data)
