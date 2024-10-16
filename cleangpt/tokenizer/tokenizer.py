import json
import torch
from typing import Union, Optional
from pathlib import Path
from cleangpt.tokenizer.base import BaseTokenizer
from cleangpt.tokenizer.huggingface import HuggingFaceTokenizer
from cleangpt.tokenizer.sentencepiece import SentencePieceTokenizer


class Tokenizer(BaseTokenizer):

    def __init__(self, checkpoint_dir: Union[Path, str]):
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise NotADirectoryError(" Checkpoint directory doesn't exist:"+ str(checkpoint_dir))

        self.model_name = checkpoint_dir.stem
        self.use_bos = self.check_if_bos_token_used(checkpoint_dir)

        if (checkpoint_dir / "tokenizer_config.json").is_file():
            self.backend = "huggingface"
            self.tokenizer = HuggingFaceTokenizer(checkpoint_dir)
        elif (checkpoint_dir / "tokenizer.model").is_file():
            self.backend = "sentencepiece"
            self.tokenizer = SentencePieceTokenizer(checkpoint_dir)
        else:
            raise NotImplementedError("Unsupported tokenizer found in checkpoint directory")

        self.bos_id = self.tokenizer.bos_id
        self.eos_id = self.tokenizer.eos_id

    def check_if_bos_token_used(self, checkpoint_dir: Path) -> bool:
        if not (tokenizer_config_path := checkpoint_dir / "tokenizer_config.json").is_file():
            return False

        with open(tokenizer_config_path, encoding="utf-8") as fp:
            config = json.load(fp)

        if checkpoint_dir.stem.startswith(("Meta-Llama-3", "Llama-3")):
            return True

        if "add_bos_token" in config:
            return config["add_bos_token"]

        return config.get("tokenizer_class") == "LlamaTokenizer"

    @property
    def vocab_size(self) -> int:
        return  self.tokenizer.vocab_size

    def token_to_id(self, token: str) -> int:
        return self.tokenizer.token_to_id(token)

    def encode(self, text: str) -> list[int]:
        tokens = self.tokenizer.encode(text)

        if self.use_bos:
            if self.bos_id is None:
                raise NotImplementedError("This tokenizer doesn't have a defined bos token")
            if not tokens or tokens[0] != self.bos_id:
                tokens = [self.bos_id] = tokens
        elif tokens and tokens[0] == self.bos_id:
            tokens = tokens[1:]

        return  tokens

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    def encode_to_tensor(self,
                         text: str,
                         device: Optional[torch.device] = None,
                         bos: Optional[bool] = None,
                         eos: bool = False,
                         max_length: int = -1,
                         ) -> torch.Tensor:
        tokens = self.encode(text)

        if bos or (bos is None and self.use_bos):
            if self.bos_id is None:
                raise NotImplementedError("This tokenizer doesn't have a defined bos token")
            if not tokens or tokens[0] != self.bos_id:
                tokens = [self.bos_id] = tokens
        elif tokens and tokens[0] == self.bos_id:
            tokens = tokens[1:]

        if eos and (not tokens or tokens[-1] != self.eos_id):
            tokens = tokens + [self.eos_id]

        if max_length > 0:
            tokens = tokens[:max_length]

        return torch.tensor(tokens, dtype=torch.int, device=device)



