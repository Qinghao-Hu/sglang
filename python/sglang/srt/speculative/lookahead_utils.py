from __future__ import annotations

from typing import TYPE_CHECKING, List, Type, Optional

import numpy as np
import torch
import triton
import triton.language as tl

from dataclasses import dataclass

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

from sglang.srt.speculative.eagle_utils import (
    assign_req_to_token_pool,
    create_flashinfer_kv_indices_triton,
)
from sglang.srt.mem_cache.memory_pool import TokenToKVPoolAllocator

# TODO (Qinghao): Update to new Eagle VerifyInput Implementation
class LookaheadVerifyInput:
    def __init__(
        self,
        draft_token: torch.Tensor,
        tree_mask: torch.Tensor,
        positions: torch.Tensor,
        retrive_index: torch.Tensor,
        retrive_cum_len: torch.Tensor,
        draft_token_num: torch.Tensor,
    ):
        self.draft_token = draft_token
        self.custom_mask = tree_mask
        self.positions = positions
        self.retrive_index = retrive_index
        self.retrive_cum_len = retrive_cum_len
        self.draft_token_num = draft_token_num
        self.draft_token_num_sum = draft_token_num.sum().item()

    def prepare_for_verify(self, batch: ScheduleBatch):
        batch.input_ids = self.draft_token
        batch.out_cache_loc = batch.alloc_token_slots(batch.input_ids.numel())
        bs = batch.seq_lens.numel()
        assign_req_to_token_pool[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + self.draft_token_num,
            batch.out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            triton.next_power_of_2(bs),
        )

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        batch_size = len(req_pool_indices)

        cum_kv_seq_len = torch.zeros((batch_size + 1,), dtype=torch.int32, device="cuda")

        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        self.qo_indptr = torch.zeros((batch_size + 1,), dtype=torch.int32, device="cuda")
        self.qo_indptr[1:] = torch.cumsum(self.draft_token_num, dim=0)

        kv_indices = torch.empty(cum_kv_seq_len[-1], dtype=torch.int32, device="cuda")

        create_flashinfer_kv_indices_triton[(batch_size,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        return kv_indices, cum_kv_seq_len, self.qo_indptr, self.custom_mask

    def verify(
        self,
        batch: ScheduleBatch,
        logits_output: torch.Tensor,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
    ) -> torch.Tensor:
        bs = self.retrive_cum_len.numel() - 1
        predict = torch.argmax(logits_output.next_token_logits, dim=-1)
        predict = torch.cat([predict, torch.full([1], -1, dtype=torch.long, device="cuda")], dim=-1)
        draft_token = torch.cat(
            [self.draft_token, torch.full([1], -1, dtype=torch.long, device="cuda")],
            dim=-1,
        )
        target_predict = predict[self.retrive_index]
        candidates = draft_token[self.retrive_index]
        accept_mask = candidates[:, 1:] == target_predict[:, :-1]
        accept_mask = (torch.cumprod(accept_mask, dim=1)).sum(dim=1)

        max_draft_len = self.retrive_index.shape[-1]
        accept_index = torch.full((bs, max_draft_len), -1, dtype=torch.long, device="cuda")
        accept_length = torch.empty((bs,), dtype=torch.int, device="cuda")
        extract_index = torch.full((bs * 2,), 0, dtype=torch.int, device="cuda")
        eagle_verify_retrive[(bs,)](
            self.retrive_index.contiguous(),
            accept_mask.contiguous(),
            self.retrive_cum_len,
            accept_index,
            accept_length,
            extract_index,
            max_draft_len,
            triton.next_power_of_2(self.draft_token_num.max().item()),
            triton.next_power_of_2(max_draft_len),
        )

        eos_token_id = batch.reqs[0].tokenizer.eos_token_id
        # TODO: check other criteria for end check
        mask_tensor = (predict[accept_index] == eos_token_id).int()
        first_true_indices = torch.argmax(mask_tensor, dim=1)
        has_true = torch.any(mask_tensor, dim=1)
        if torch.any(mask_tensor):
            batch_size, seq_length = accept_index.shape
            range_vec = (
                torch.arange(seq_length, device="cuda").unsqueeze(0).repeat(batch_size, 1)
            )  # shape: (batch_size, seq_length)
            threshold = first_true_indices.unsqueeze(1) + 1  # shape: (batch_size, 1)
            mask = (range_vec >= threshold) & has_true.unsqueeze(1)  # shape: (batch_size, seq_length)
            accept_index[mask] = -1

        accept_length = (accept_index != -1).sum(dim=1)

        accept_index_flatten = accept_index[accept_index != -1]

        assign_req_to_token_pool[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + accept_length,
            batch.out_cache_loc[accept_index_flatten],
            batch.req_to_token_pool.req_to_token.shape[1],
            triton.next_power_of_2(bs),
        )
        batch.seq_lens.add_(accept_length)  # TODO: mcheck the case for normal decoding
        batch.seq_lens_sum = batch.seq_lens.sum().item()

        evict_mask = torch.full_like(self.draft_token, True, dtype=torch.bool)
        evict_mask[accept_index_flatten] = False
        mem_need_free_idx = batch.out_cache_loc[evict_mask]
        token_to_kv_pool_allocator.free(mem_need_free_idx)

        last_verified_ids = []
        accept_token_bs = []
        for i in range(bs):
            req = batch.reqs[i]
            accept_ids = torch.where(accept_index[i] != -1)[0]
            accept_token = predict[accept_index[i][accept_ids]]
            accept_token_cpu = accept_token.tolist()
            req.output_ids.extend(accept_token_cpu)
            accept_token_bs.append(accept_token)
            req.check_finished()
            # need to append the token for scheduler process_batch_result_decode to work
            last_verified_ids.append(req.output_ids[-1])

        verified_id = predict[accept_index_flatten]
        verified_id_cpu = verified_id.tolist()

        last_verified_ids = torch.tensor(last_verified_ids, device="cuda")
        logits_output.next_token_logits = logits_output.next_token_logits[accept_index_flatten]
        return logits_output, last_verified_ids, accept_length.sum().item()

    def merge_batch(self, spec_info: LookaheadVerifyInput):
        return


@triton.jit
def eagle_verify_retrive(
    retrive_index,
    accept_mask,
    retrive_cum_len,
    accept_index,
    accept_length,
    extract_index,
    max_len: tl.constexpr,
    draft_token_num: tl.constexpr,
    max_len_upper: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    retrive_end = tl.load(retrive_cum_len + pid + 1)
    retrive_start = tl.load(retrive_cum_len + pid)
    retrive_len = retrive_end - retrive_start
    accept_ptr = accept_mask + retrive_start
    accept_offset = tl.arange(0, draft_token_num)
    accept_load_mask = accept_offset < retrive_len
    accept_len_list = tl.load(accept_ptr + accept_offset, mask=accept_load_mask, other=-1)

    accept_len = tl.max(accept_len_list)
    max_index = tl.argmax(accept_len_list, axis=0, tie_break_left=True)
    # triton is not support argmax with tie_break_right, so I need implement it by some way
    mask_max = accept_len_list == accept_len

    count_mask = tl.full(shape=[draft_token_num], value=0, dtype=tl.int32)
    count = tl.sum(tl.where(mask_max, 1, count_mask))
    if count > 1:
        index = tl.arange(0, draft_token_num)
        mask_left = index != max_index
        remained_index = tl.where(mask_max and mask_left, index, 0)
        max_index = tl.max(remained_index)

    tl.store(accept_length + pid, accept_len)
    retrive_index_ptr = retrive_index + (retrive_start + max_index) * max_len
    retrive_offset = tl.arange(0, max_len_upper)
    retrive_load_mask = retrive_offset < accept_len + 1
    data = tl.load(retrive_index_ptr + retrive_offset, mask=retrive_load_mask)

    tl.store(accept_index + pid * max_len + retrive_offset, data, mask=retrive_load_mask)

    extract_load_ptr = accept_index + pid * max_len + accept_len
    if accept_len == max_len - 1:
        extract_data = tl.load(extract_load_ptr - 1)
        tl.store(extract_index + pid * 2, extract_data)
        extract_data = tl.load(extract_load_ptr)
        tl.store(extract_index + pid * 2 + 1, extract_data)

    else:
        extract_data = tl.load(extract_load_ptr)
        tl.store(extract_index + pid * 2, extract_data)
