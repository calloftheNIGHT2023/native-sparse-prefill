"""Fused causal masking and small-k ranking over materialized query tiles.

Still scores every causal candidate: quadratic arithmetic, query-tiled storage.
Equal-score ties select the lowest key index. torch.topk does not guarantee that
tie rule, so equality of support is required only at a strict selection boundary.
"""
import math
import torch
import triton
import triton.language as tl
from chunked_topk_attention import _validate
from triton_selected_attention import TritonTopKAttention, triton_selected_attention


@triton.jit
def _rank_tile(SCORES, IDS, N: tl.constexpr, FIRST: tl.constexpr, C: tl.constexpr,
               PREFIX: tl.constexpr, S: tl.constexpr, LOCAL: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0); group = tl.program_id(1)
    position = FIRST + row
    keys = tl.arange(0, BN)
    scores = tl.load(SCORES + (group * C + row) * PREFIX + keys, keys < PREFIX, -float('inf')).to(tl.float32)
    scores = tl.where((keys < PREFIX) & (keys <= position - LOCAL), scores, -float('inf'))
    for offset in tl.static_range(LOCAL):
        local_id = position - offset
        tl.store(IDS + (group * N + position) * S + offset, tl.where(local_id >= 0, local_id, -1))
    for rank in tl.static_range(S - LOCAL):
        candidate = tl.argmax(scores, 0, tie_break_left=True)
        value = tl.max(scores, 0)
        valid = value != -float('inf')
        tl.store(IDS + (group * N + position) * S + LOCAL + rank, tl.where(valid, candidate, -1))
        scores = tl.where(keys == candidate, -float('inf'), scores)


@torch.no_grad()
def select_triton_topk(q, k, budget=8, local=2, query_chunk=1024):
    _validate(q, k)
    if not q.is_cuda or q.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError('CUDA FP32/BF16/FP16 required')
    if not 1 <= local <= budget <= 32 or query_chunk < 1 or q.shape[1] > 32768:
        raise ValueError('Require budget <=32, length <=32768 and valid local/chunk')
    groups, length, dim = q.shape
    slots = min(budget, length); local_slots = min(local, slots)
    ids = torch.empty(groups, length, slots, device=q.device, dtype=torch.long)
    scaled = k / math.sqrt(dim)
    for first in range(0, length, query_chunk):
        end = min(first + query_chunk, length)
        score = torch.bmm(q[:, first:end], scaled[:, :end].transpose(1, 2))
        _rank_tile[(end-first, groups)](score, ids, length, first, end-first, end,
            slots, local_slots, triton.next_power_of_2(end), num_warps=4 if end <= 4096 else 8)
    return ids


class TritonRankTopKAttention(TritonTopKAttention):
    def forward(self, qkv):
        batch, length, three, heads, dim = qkv.shape
        if three != 3:
            raise ValueError('Expected [B,N,3,H,D]')
        q, k, v = [z.permute(0, 2, 1, 3).reshape(batch*heads, length, dim) for z in qkv.unbind(2)]
        ids = select_triton_topk(q, k, self.k, self.local, self.query_chunk)
        output = triton_selected_attention(q, k, v, ids, self.dropout_p if self.training else 0.)
        return output.reshape(batch, heads, length, dim).permute(0, 2, 1, 3)
