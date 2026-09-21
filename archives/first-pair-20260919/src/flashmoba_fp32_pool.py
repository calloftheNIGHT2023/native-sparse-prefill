# Adapted from FlashMoBA mean_pool_kernel, copyright (c) 2025 FlashMoBA Team.
# Upstream commit 39d9ac043b271d046a2181a9991e99a26b67bca1; see third_party license.
# Ordinary selective-precision repair, not a proposed original research method.
import triton
import triton.language as tl

@triton.jit
def fp32_mean_pool_kernel(input_ptr,output_ptr,HEAD_DIM:tl.constexpr,POOL_BLOCK_SIZE:tl.constexpr,
        cu_seqlens_input,cu_seqlens_output,input_stride_row,input_stride_head,
        output_stride_row,output_stride_head,kBlockN:tl.constexpr):
    n_block=tl.program_id(0);bidb=tl.program_id(1);bidh=tl.program_id(2)
    seq_start=tl.load(cu_seqlens_input+bidb);seq_end=tl.load(cu_seqlens_input+bidb+1)
    block_start_row=seq_start+n_block*POOL_BLOCK_SIZE
    if seq_end<=block_start_row:return
    actual_block_size=tl.minimum(POOL_BLOCK_SIZE,seq_end-block_start_row)
    offsets_d=tl.arange(0,HEAD_DIM);acc=tl.zeros([HEAD_DIM],dtype=tl.float32)
    for block_k_start in range(0,actual_block_size,kBlockN):
        offsets_k=block_k_start+tl.arange(0,kBlockN)
        offsets=(block_start_row+offsets_k[:,None])*input_stride_row.to(tl.int64)+bidh*input_stride_head.to(tl.int64)+offsets_d[None,:]
        inp=tl.load(input_ptr+offsets,mask=(offsets_k<actual_block_size)[:,None],other=0.0)
        acc+=tl.sum(inp.to(tl.float32),axis=0)
    output_start=tl.load(cu_seqlens_output+bidb)
    offsets=(output_start+n_block)*output_stride_row.to(tl.int64)+bidh*output_stride_head.to(tl.int64)+offsets_d
    tl.store(output_ptr+offsets,acc/actual_block_size)
