from typing import List, Union
from embedbase import get_app
from embedbase.database.memory_db import MemoryDatabase
from embedbase.embedding.base import Embedder
import uvicorn
import llama_cpp
from llama_cpp import Llama
 
 
class LlamaEmbedder(Embedder):
    EMBEDDING_MODEL = "ggml-vicuna-7b-4bit-rev1.bin"
 
    def __init__(
        self, model: str = EMBEDDING_MODEL, **kwargs
    ):
        super().__init__(**kwargs)
        self.model = Llama(model_path=model, embedding=True)
        self.model.create_embedding("Hello world!")
 
    @property
    def dimensions(self) -> int:
        """
        Return the dimensions of the embeddings
        :return: dimensions of the embeddings
        """
        return llama_cpp.llama_n_embd(self.model.ctx)
 
    def is_too_big(self, text: str) -> bool:
        """
        Check if text is too big to be embedded,
        delegating the splitting UX to the caller
        :param text: text to check
        :return: True if text is too big, False otherwise
        """
        return len(text) > self.model.params.n_ctx
 
    async def embed(self, data: Union[List[str], str]) -> List[List[float]]:
        """
        Embed a list of texts
        :param texts: list of texts
        :return: list of embeddings
        """
        return [self.model.embed(e) for e in data]
 
embedder = LlamaEmbedder("ggml-vicuna-7b-4bit-rev1.bin")
app = (
    get_app()
    .use_embedder(embedder)
    .use_db(MemoryDatabase(dimensions=embedder.dimensions))
    .run()
)
 
if __name__ == "__main__":
    uvicorn.run("main:app")
