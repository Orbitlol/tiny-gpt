from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 8000
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    dropout: float = 0.1
    bias: bool = False
