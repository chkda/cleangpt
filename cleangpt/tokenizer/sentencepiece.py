from sentencepiece import SentencePieceProcessor
from pathlib import Path
from cleangpt.tokenizer.base import BaseTokenizer


class SentencePieceTokenizer(BaseTokenizer):

    def __init__(self, checkpoint_dir: Path):
        vocabulary_path = checkpoint_dir / "tokenizer.model"
        self.tokenizer = SentencePieceProcessor(model_file=str(vocabulary_path))
        self.bos_id = self.tokenizer.bos_id()
        self.eos_id = self.tokenizer.eos_id()

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size()

    def token_to_id(self, token: str) -> int:
        id_ = self.tokenizer.PieceToId(token)
        if id_ is None:
            raise ValueError(token + " not found in the collection.")
        return  id_

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.Encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.Decode(tokens)