import json
from pathlib import Path
from tokenizers import Tokenizer as HFTokenizer

from cleangpt.tokenizer.base import BaseTokenizer
from cleangpt.utils import fix_and_load_json

class HuggingFaceTokenizer(BaseTokenizer):

    def __init__(self, checkpoint_dir: Path):
        vocabulary_path = checkpoint_dir / "tokenizer.json"
        self.tokenizer = HFTokenizer.from_file(str(vocabulary_path))

        self.bos_id = None
        self.eos_id = None

        self._load_special_tokens(checkpoint_dir)
        self.apply_decoding_fix = self._check_decoding_fix(checkpoint_dir)

    def _load_special_tokens(self, checkpoint_dir: Path):
        if (special_tokens_path := checkpoint_dir / "tokenizer_config.json").is_file():
            with open(special_tokens_path, encoding="utf-8") as fp:
                config = json.load(fp)

            bos_token = config.get("bos_token")
            eos_token = config.get("eos_token")

            if bos_token is not None and isinstance(bos_token, dict):
                bos_token = bos_token.get("content")
            if eos_token is not None and isinstance(eos_token, dict):
                eos_token = eos_token.get("content")

            self.bos_id = self.token_to_id(bos_token) if bos_token is not None else None
            self.eos_id = self.token_to_id(eos_token) if eos_token is not None else None

        if (special_tokens_path := checkpoint_dir / "generation_config.json").is_file():
            try:
                with open(special_tokens_path, encoding="utf-8") as fp:
                    config = json.load(fp)
            except json.JSONDecodingError:
                with open(special_tokens_path, encoding="utf-8") as fp:
                    json_string = fp.read()
                    config = fix_and_load_json(json_string)
            
            if self.bos_id is None:
                self.bos_id = config.get("bos_token_id")
            if self.eos_id is None:
                self.eos_id = config.get("eos_token_id")

    def _check_decoding_fix(self, checkpoint_dir: Path) -> bool:
        if (config_path := checkpoint_dir/ "tokenizer_config.json").is_file():
            with open(config_path, encoding="utf-8") as fp:
                return "LlamaTokenizer" in json.load(fp)["tokenizer_class"]
        return False

    def token_to_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(token + " not found in the collection.")
        return token_id

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size(with_added_tokens=False)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.token_to_id(text)

    def decode(self, tokens: list[int]) -> str:
        if len(tokens) == 1 and self.apply_decoding_fix:
            dummy_token_id = 33
            dummy_token = self.tokenizer.decode([dummy_token_id])
            return self.tokenizer.decode([dummy_token] + tokens)[len(dummy_token):]
        return self.tokenizer.decode(tokens)
        