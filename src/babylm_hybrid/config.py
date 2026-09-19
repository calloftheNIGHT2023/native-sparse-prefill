"""Explicit configuration for D/E correctness and later bounded baseline runs."""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class HybridConfig:
    vocab_size: int = 97
    hidden_size: int = 32
    intermediate_size: int = 64
    layer_types: tuple = ("gdn", "gdn", "gdn", "global")
    tie_word_embeddings: bool = True
    norm_eps: float = 1e-6
    initializer_range: float = 0.02
    gdn_key_heads: int = 1
    gdn_value_heads: int = 3
    gdn_key_head_dim: int = 8
    gdn_value_head_dim: int = 8
    gdn_conv_kernel: int = 4
    global_q_heads: int = 4
    global_kv_heads: int = 2
    global_head_dim: int = 8
    global_rotary_dim: int = 4
    rope_theta: float = 1e7
    index_query_heads: int = 2
    index_head_dim: int = 8
    index_rotary_dim: int = 4
    block_size: int = 4
    selected_complete_blocks: int = 2
    index_score_scale: float = 1.0

    def __post_init__(self):
        positive = ["vocab_size", "hidden_size", "intermediate_size", "gdn_key_heads",
                    "gdn_value_heads", "gdn_key_head_dim", "gdn_value_head_dim",
                    "gdn_conv_kernel", "global_q_heads", "global_kv_heads",
                    "global_head_dim", "index_query_heads", "index_head_dim",
                    "block_size", "selected_complete_blocks"]
        for key in positive:
            if not isinstance(getattr(self, key), int) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.gdn_value_heads % self.gdn_key_heads or self.global_q_heads % self.global_kv_heads:
            raise ValueError("GDN/GQA grouped head counts must divide evenly")
        if not self.layer_types or any(t not in ("gdn", "global") for t in self.layer_types):
            raise ValueError("Nonempty explicit gdn/global layer types required")
        for rot, dim in [(self.global_rotary_dim, self.global_head_dim),
                         (self.index_rotary_dim, self.index_head_dim)]:
            if rot <= 0 or rot % 2 or rot > dim:
                raise ValueError("Rotary dimensions must be positive, even, <= head size")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_candidate(cls, candidate):
        m, q = candidate["model_candidate"], candidate["qsa_candidate"]
        return cls(vocab_size=m["vocab_size"], hidden_size=m["hidden_size"],
                   intermediate_size=m["intermediate_size"], layer_types=tuple(m["layers"]),
                   tie_word_embeddings=m["tie_word_embeddings"],
                   gdn_key_heads=m["gdn_key_heads"], gdn_value_heads=m["gdn_value_heads"],
                   gdn_key_head_dim=m["gdn_key_head_dim"], gdn_value_head_dim=m["gdn_value_head_dim"],
                   gdn_conv_kernel=m["gdn_conv_kernel"], global_q_heads=m["global_q_heads"],
                   global_kv_heads=m["global_kv_heads"], global_head_dim=m["global_head_dim"],
                   global_rotary_dim=m["global_rotary_dim"], rope_theta=m["rope_theta"],
                   index_query_heads=q["index_query_heads"], index_head_dim=q["index_head_dim"],
                   index_rotary_dim=q["index_rotary_dim"], block_size=q["block_size"],
                   selected_complete_blocks=q["selected_complete_blocks"], index_score_scale=q["score_scale"])

