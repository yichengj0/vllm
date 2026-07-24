<!-- Draft PR description for this branch. Paste into the PR body when opening; delete this file from the branch first. FlashInfer PR number below is TBD until it is opened. -->

## Purpose

On SM120 and SM121 (consumer Blackwell and DGX Spark), the FlashInfer backend cannot serve models that use attention sinks. `supports_sink()` allows sinks only on the SM100 trtllm-gen path, so on SM12x a sink model falls back to the Triton backend. The FlashInfer decode wrappers also copy `seq_lens` to the host inside `plan()`, which forces a sync on every step and stops the scheduler from running ahead.

FlashInfer's XQA decode kernel already runs on SM12x with attention sinks, sliding window, and speculative-decode masks, and it reads `seq_lens` on the device, so it needs no host sync. This PR routes SM12x decode and speculative decode to XQA and lets the FlashInfer backend serve sink models there.

Changes:

- Enable the XQA decode kernel on SM120/121. Prefill stays on the native FlashInfer kernel. XQA is built from source, so the cubin-download requirement now applies only to the trtllm-gen path.
- Allow attention sinks on SM120/121. XQA applies the sink during decode, and the native FlashInfer prefill kernel applies it during prefill.
- Support speculative decode by passing XQA an explicit packed causal draft mask, cached per draft length so it is safe under CUDA graphs.
- Relax the metadata-builder guard so a batch with sinks is allowed when decode runs on XQA even though prefill runs on the native kernel.

This needs a FlashInfer build with the XQA speculative-decode sliding-window fix (flashinfer-ai/flashinfer#4137). Plain sink models work with released FlashInfer. NVFP4 KV cache during prefill is still unsupported on SM12x.

## Test Plan

- FlashInfer XQA kernel tests on SM120 and SM121, covering the sink, sliding-window, and speculative-decode-mask cases at the target attention shape (head_dim 128, 32 query heads, 2 KV heads).
- Byte-compare the draft mask this PR builds against the FlashInfer kernel-test reference.
- Kernel benchmarks at the target shape against the Triton backend on RTX PRO 6000 and GB10.
- End-to-end serving of a sink model with speculative decode (pending).

## Test Result

- Kernel tests pass on RTX 5080, RTX PRO 6000, and GB10. The draft mask is byte-identical to the reference across every tested draft length.
- At the target shape, XQA is faster than the Triton backend on both RTX PRO 6000 and GB10, roughly 2 to 4 times on the speculative-decode steps, and turning on the sink adds under a microsecond.
- End-to-end serving results are pending.
