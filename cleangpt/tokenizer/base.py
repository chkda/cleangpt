from pathlib import Path
from abc import ABC, abstractmethod

class BaseTokenizer(ABC):

    @abstractmethod
    def __init__(self, checkpoint_dir:Path):
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @abstractmethod
    def token_to_id(self, token: str) -> int:
        pass

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        pass

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        pass