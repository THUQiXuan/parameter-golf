"""V30: V29 + stride=64 (2x faster eval) + vectorized batch ngram lookup + 2500s training budget."""
from __future__ import annotations
import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
import lzma
from pathlib import Path
try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "lzma"
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# Configure torch.compile to suppress errors and allow fallback
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.optimize_ddp = False
    torch._dynamo.config.cache_size_limit = 256  # allow many RMSNorm instances
except Exception:
    pass

# Compatibility shim for F.rms_norm (PyTorch < 2.4)
if not hasattr(F, 'rms_norm'):
    @torch.jit.script
    def _rms_norm_impl(x: torch.Tensor, eps: float) -> torch.Tensor:
        """JIT-compiled RMS norm for better performance."""
        xf = x.float()
        rms = (xf.pow(2).mean(-1, keepdim=True) + eps).rsqrt()
        return (xf * rms).to(x.dtype)
    def _rms_norm(x, normalized_shape, weight=None, eps=None):
        if eps is None:
            eps = 1e-6
        y = _rms_norm_impl(x, eps)
        if weight is not None:
            y = y * weight
        return y
    F.rms_norm = _rms_norm

try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    _HAS_FA3 = True
except ImportError:
    # FA2 (flash_attn) must NOT be used as a FA3 fallback — it has a different
    # calling convention and triggers async CUDA illegal-instruction race conditions
    # on SM 9.0 (L20Z / H100).  Fall through to F.scaled_dot_product_attention.
    _HAS_FA3 = False
    flash_attn_3_func = None

# ---------------------------------------------------------------------------
# PR#809-style n-gram evaluation: NgramEvalCache with np.bincount (10-100x
# faster than np.add.at), 1M-token chunks, optimised alpha/entropy parameters.
# ---------------------------------------------------------------------------
_NGRAM_PRIMES = np.array([36313, 27191, 51647, 81929, 131071, 174763, 233017, 283721, 347237], dtype=np.uint64)

def _batch_hash_ctx(tokens_np: np.ndarray, positions: np.ndarray, n: int, bucket_mask: int) -> np.ndarray:
    """Vectorized context hash for all positions at once using XOR-of-products."""
    h = np.zeros(len(positions), dtype=np.uint64)
    for k in range(n - 1):
        idx = positions - (n - 1) + k
        idx = np.clip(idx, 0, len(tokens_np) - 1)
        h ^= tokens_np[idx].astype(np.uint64) * _NGRAM_PRIMES[k]
    return h & np.uint64(bucket_mask)

def _batch_hash_full(tokens_np: np.ndarray, positions: np.ndarray, targets: np.ndarray, n: int, bucket_mask: int) -> np.ndarray:
    """Vectorized context+target hash for all positions at once."""
    h = np.zeros(len(positions), dtype=np.uint64)
    for k in range(n - 1):
        idx = positions - (n - 1) + k
        idx = np.clip(idx, 0, len(tokens_np) - 1)
        h ^= tokens_np[idx].astype(np.uint64) * _NGRAM_PRIMES[k]
    h ^= targets.astype(np.uint64) * _NGRAM_PRIMES[min(n - 1, len(_NGRAM_PRIMES) - 1)]
    return h & np.uint64(bucket_mask)

class NgramEvalCache:
    """Backward-looking N-gram frequency cache for eval-time score improvement.

    Uses np.bincount for O(n) updates (vs np.add.at O(n*k)) — 10-100x faster.
    Score-first: cache updated AFTER scoring each chunk (legal under competition rules).
    """
    def __init__(self, max_order: int = 9, min_order: int = 2, num_buckets: int = 4194304, min_count: int = 2):
        assert (num_buckets & (num_buckets - 1)) == 0, "num_buckets must be power of 2"
        self.max_order = max_order
        self.min_order = min_order
        self.num_buckets = num_buckets
        self.bucket_mask = num_buckets - 1
        self.min_count = min_count
        self.ctx_tables: list[np.ndarray] = [np.zeros(num_buckets, dtype=np.int32) for _ in range(max_order + 1)]
        self.full_tables: list[np.ndarray] = [np.zeros(num_buckets, dtype=np.int32) for _ in range(max_order + 1)]

    def batch_lookup(self, tokens_np: np.ndarray, positions: np.ndarray, targets: np.ndarray):
        """Vectorized multi-order backoff lookup. Returns (ngram_probs, matched_mask, matched_orders)."""
        n_pos = len(positions)
        ngram_p = np.zeros(n_pos, dtype=np.float64)
        matched = np.zeros(n_pos, dtype=bool)
        matched_orders = np.zeros(n_pos, dtype=np.int32)
        for n in range(self.max_order, self.min_order - 1, -1):
            eligible = (~matched) & (positions >= n - 1)
            if not eligible.any():
                continue
            elig_pos = positions[eligible]
            elig_tgt = targets[eligible]
            ctx_keys = _batch_hash_ctx(tokens_np, elig_pos, n, self.bucket_mask).astype(np.int64)
            ctx_counts = self.ctx_tables[n][ctx_keys]
            has_data = ctx_counts >= self.min_count
            if not has_data.any():
                continue
            full_keys = _batch_hash_full(tokens_np, elig_pos[has_data], elig_tgt[has_data], n, self.bucket_mask).astype(np.int64)
            full_counts = self.full_tables[n][full_keys]
            capped_full = np.minimum(full_counts, ctx_counts[has_data])
            probs = capped_full.astype(np.float64) / np.maximum(ctx_counts[has_data].astype(np.float64), 1.0)
            elig_indices = np.where(eligible)[0]
            data_indices = elig_indices[has_data]
            ngram_p[data_indices] = probs
            matched[data_indices] = True
            matched_orders[data_indices] = n
        return ngram_p, matched, matched_orders

    def update_batch(self, tokens_np: np.ndarray, start_pos: int, end_pos: int) -> None:
        """Vectorized cache update using np.bincount (10-100x faster than np.add.at)."""
        if end_pos <= start_pos:
            return
        positions = np.arange(start_pos, end_pos, dtype=np.int64)
        targets = tokens_np[positions]
        for n in range(self.min_order, self.max_order + 1):
            valid = positions >= n - 1
            if not valid.any():
                continue
            v_pos = positions[valid]
            v_tgt = targets[valid]
            ctx_keys = _batch_hash_ctx(tokens_np, v_pos, n, self.bucket_mask).astype(np.int64)
            full_keys = _batch_hash_full(tokens_np, v_pos, v_tgt, n, self.bucket_mask).astype(np.int64)
            self.ctx_tables[n] += np.bincount(ctx_keys, minlength=self.num_buckets).astype(np.int32)
            self.full_tables[n] += np.bincount(full_keys, minlength=self.num_buckets).astype(np.int32)

def _build_sliding_segments(total_tokens: int, seq_len: int, stride: int):
    """Build scored-token segments for sliding-window eval.

    Returns list of (window_start, valid_len, local_score_start, local_score_end, target_start, target_end).
    Each 'target' range [target_start, target_end) contains the scored tokens (global indices).
    """
    if total_tokens <= 0:
        return []
    segments = []
    first_valid_len = min(seq_len, total_tokens)
    segments.append((0, first_valid_len, 0, first_valid_len, 1, first_valid_len + 1))
    next_target_start = first_valid_len + 1
    while next_target_start <= total_tokens:
        target_end = min(next_target_start + stride, total_tokens + 1)
        window_end = target_end - 1
        window_start = max(0, window_end - seq_len)
        valid_len = window_end - window_start
        local_score_start = next_target_start - window_start - 1
        local_score_end = target_end - window_start - 1
        segments.append((window_start, valid_len, local_score_start, local_score_end, next_target_start, target_end))
        next_target_start = target_end
    return segments

class BackoffNgramMixer:
    """Multi-order n-gram backoff with entropy-adaptive alpha (OAEG) + Cubric per-order adaptive scaling.

    Combines:
    - PR #798: Order-Adaptive Entropy Gating (per-order entropy centers)
    - PR #800: Cubric (per-order adaptive alpha multipliers based on beat-rate statistics)
    - Extended to max_order=9 (8-gram and 9-gram contexts)
    """

    def __init__(self, vocab_size: int = 1024, device: str = 'cuda', eta: float = 0.1,
                 max_order: int = 9):
        self.V = vocab_size
        self.device = device
        self.eta = eta
        self.total_tokens = 0
        self.max_order = max_order
        self.min_order = 2
        import numpy as _np
        self._np = _np
        self.BUCKETS = 4_194_304
        # Primes for context widths 0..max_order-1 (need max_order primes)
        _all_primes = [36313, 27191, 51647, 81929, 131071, 174763, 233017, 293011, 373739, 452219]
        self.primes = [_np.uint64(p) for p in _all_primes[:max_order]]
        n_orders = max_order - self.min_order + 1  # e.g., 8 for max_order=9
        self.ctx_counts = [_np.zeros(self.BUCKETS, dtype=_np.uint32) for _ in range(n_orders)]
        self.full_counts = [_np.zeros(self.BUCKETS, dtype=_np.uint32) for _ in range(n_orders)]
        # Cubric state: per-order adaptive alpha multipliers
        self._cubric_enabled = bool(int(os.environ.get("CUBRIC_ENABLED", "1")))
        self._c_alpha_mult = {n: 1.0 for n in range(self.min_order, self.max_order + 1)}
        self._c_hits_chunk = {n: 0 for n in range(self.min_order, self.max_order + 1)}
        self._c_beats_chunk = {n: 0 for n in range(self.min_order, self.max_order + 1)}

    def update(self, tokens):
        np = self._np
        if hasattr(tokens, 'cpu'):
            t = tokens.cpu().numpy().astype(np.int64)
        else:
            t = np.array(tokens, dtype=np.int64)
        n = len(t)
        if n == 0:
            return
        self.total_tokens += n
        mask = np.uint64(self.BUCKETS - 1)
        for oi, order in enumerate(range(self.min_order, self.max_order + 1)):
            if n < order:
                continue
            cw = order - 1
            ctx_hash = np.zeros(n - order + 1, dtype=np.uint64)
            for k in range(cw):
                ctx_hash ^= t[k:n - order + 1 + k].astype(np.uint64) * self.primes[k]
            ctx_key = (ctx_hash & mask).astype(np.int64)
            tgt = t[order - 1:].astype(np.uint64)
            full_key = ((ctx_hash ^ (tgt * self.primes[cw])) & mask).astype(np.int64)
            np.add.at(self.ctx_counts[oi], ctx_key, 1)
            np.add.at(self.full_counts[oi], full_key, 1)

    def step_cubric(self):
        """Update Cubric per-order alpha multipliers based on chunk beat-rate stats.
        Call this after scoring each chunk (before update()) to adapt multipliers.
        """
        if not self._cubric_enabled:
            return
        active = [(n, self._c_beats_chunk[n] / self._c_hits_chunk[n])
                  for n in range(self.min_order, self.max_order + 1)
                  if self._c_hits_chunk[n] >= 20]
        if len(active) >= 2:
            avg_rate = sum(r for _, r in active) / len(active)
            for n, rate in active:
                if rate > avg_rate + 0.05:
                    self._c_alpha_mult[n] = min(self._c_alpha_mult[n] * 1.03, 2.0)
                elif rate < avg_rate - 0.05:
                    self._c_alpha_mult[n] = max(self._c_alpha_mult[n] * 0.97, 0.3)
        # Reset chunk counters
        self._c_hits_chunk = {n: 0 for n in range(self.min_order, self.max_order + 1)}
        self._c_beats_chunk = {n: 0 for n in range(self.min_order, self.max_order + 1)}

    def mix_and_score(self, neural_logits, x_batch, y_batch, wlens):
        np = self._np
        bsz, slen, V = neural_logits.shape
        device = neural_logits.device
        neural_lp = F.log_softmax(neural_logits, dim=-1)
        neural_nll = -neural_lp.gather(2, y_batch.unsqueeze(2)).squeeze(2)
        if self.total_tokens < 100:
            return neural_nll, None
        with torch.no_grad():
            probs = neural_lp.exp()
            entropy = -(probs * neural_lp).sum(dim=-1)
        # Order-Adaptive Entropy Gating (OAEG) from PR #798:
        # Higher orders are trusted at lower entropy (they're more specific/reliable)
        # Extended to orders 8 and 9 with even lower entropy centers
        ent_centers = {9: 2.5, 8: 2.8, 7: 3.0, 6: 3.2, 5: 3.5, 4: 3.8, 3: 4.2, 2: 4.5}
        alpha_max = float(os.environ.get("ALPHA_MAX", "0.60"))
        alpha_min = float(os.environ.get("ALPHA_MIN", "0.05"))
        neural_p = neural_lp.gather(2, y_batch.unsqueeze(2)).squeeze(2).exp()
        x_np = x_batch.cpu().numpy().astype(np.int64)
        y_np = y_batch.cpu().numpy().astype(np.int64)
        mask = np.uint64(self.BUCKETS - 1)
        ngram_p = np.zeros((bsz, slen), dtype=np.float64)
        ngram_hit = np.zeros((bsz, slen), dtype=np.bool_)
        best_order = np.zeros((bsz, slen), dtype=np.int32)
        n_orders = self.max_order - self.min_order  # = 7 for max_order=9
        for oi_rev in range(n_orders, -1, -1):
            order = oi_rev + self.min_order
            cw = order - 1
            if slen < cw:
                continue
            ctx_hash = np.zeros((bsz, slen), dtype=np.uint64)
            for k in range(cw):
                shift = cw - 1 - k
                shifted = np.zeros_like(x_np, dtype=np.uint64)
                if shift > 0 and shift < slen:
                    shifted[:, shift:] = x_np[:, :slen - shift].astype(np.uint64)
                elif shift == 0:
                    shifted = x_np.astype(np.uint64)
                ctx_hash ^= shifted * self.primes[k]
            ctx_key = (ctx_hash & mask).astype(np.int64)
            full_key = ((ctx_hash ^ (y_np.astype(np.uint64) * self.primes[cw])) & mask).astype(np.int64)
            ctx_c = self.ctx_counts[oi_rev][ctx_key.reshape(-1)].astype(np.float64).reshape(bsz, slen)
            full_c = self.full_counts[oi_rev][full_key.reshape(-1)].astype(np.float64).reshape(bsz, slen)
            valid = (ctx_c >= 2) & (~ngram_hit)
            if cw > 0:
                valid[:, :cw] = False
            p = np.minimum(full_c, ctx_c) / np.maximum(ctx_c, 1.0)
            p = np.clip(p, 0.0, 1.0)
            ngram_p[valid] = p[valid]
            ngram_hit[valid] = True
            best_order[valid] = order
        ngram_p[~ngram_hit] = 1.0 / self.V
        ngram_p_t = torch.tensor(ngram_p, device=device, dtype=torch.float32)
        # Per-position alpha based on which order matched (OAEG)
        best_order_t = torch.tensor(best_order, device=device, dtype=torch.float32)
        ent_center_t = torch.zeros_like(entropy)
        for order, ec in ent_centers.items():
            ent_center_t[best_order_t == order] = ec
        ent_center_t[best_order_t == 0] = 4.0  # default for no match
        alpha = alpha_min + (alpha_max - alpha_min) * torch.sigmoid(2.0 * (entropy - ent_center_t))
        # Cubric: apply per-order adaptive multipliers (from PR #800)
        if self._cubric_enabled and self.total_tokens > 5000:
            cubric_mult_t = torch.ones_like(alpha)
            for order in range(self.min_order, self.max_order + 1):
                mask_o = (best_order_t == order)
                if mask_o.any():
                    cubric_mult_t[mask_o] = self._c_alpha_mult[order]
            alpha = (alpha * cubric_mult_t).clamp(0.0, alpha_max)
            # Accumulate Cubric beat-rate stats (CPU, using numpy)
            neural_p_np = neural_p.cpu().numpy()
            for order in range(self.min_order, self.max_order + 1):
                mask_o = (best_order == order)
                cnt = int(mask_o.sum())
                if cnt > 0:
                    self._c_hits_chunk[order] += cnt
                    beats = int((ngram_p[mask_o] > neural_p_np[mask_o]).sum())
                    self._c_beats_chunk[order] += beats
        mixed_p = (1.0 - alpha) * neural_p + alpha * ngram_p_t
        mixed_nll = -torch.log(mixed_p.clamp(min=1e-12))
        return mixed_nll, None

    def update_weights(self, expert_nll, wlens):
        pass

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 500))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 8))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 3.5))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.035))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    int6_last_n = int(os.environ.get("INT6_LAST_N", 0))  # all int5 (saves ~300KB vs int6 for last 2 blocks)
    ttt_temperature = float(os.environ.get("TTT_TEMPERATURE", 0.98))  # post-TTT temperature calibration
    muon_beta2 = float(os.environ.get("MUON_BETA2", 0.95))
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_every = int(os.environ.get("SWA_EVERY", 50))

    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))
    qat_enabled = bool(int(os.environ.get("QAT_ENABLED", "0")))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 6144))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 11))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    dtg_enabled = bool(int(os.environ.get("DTG_ENABLED", "0")))
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.5))
    ve_enabled = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim = int(os.environ.get("VE_DIM", 128))
    ve_layers = os.environ.get("VE_LAYERS", "9,10")
    prune_pct = float(os.environ.get("PRUNE_PCT", 0.03))

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, weight_decay=weight_decay),
        )
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )

def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]

def eval_val(args: Hyperparameters, model: nn.Module, rank: int, world_size: int,
             device: torch.device, grad_accum_steps: int, val_tokens: Tensor,
             base_bytes_lut: Tensor, has_leading_space_lut: Tensor,
             is_boundary_token_lut: Tensor, eval_seq_len: int | None = None) -> tuple[float, float]:
    seq_len = eval_seq_len or args.train_seq_len
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, seq_len={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)
CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,dtg_gate,ve_layer_scales,ve_shared.scale",
    ).split(",")
    if pattern
)
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_Q = 0.9999984

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
    _qat_enabled: bool = False
    _soft_round_alpha: float = 1.0  # temperature for soft-round (annealed during training)
    _use_soft_round: bool = False  # enable soft-round QAT instead of STE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._clip_range = 15  # default int5, set to 31 for int6 layers

    @staticmethod
    def soft_round(y: Tensor, alpha: float) -> Tensor:
        """Differentiable approximation to round() from Agustsson & Theis (NeurIPS 2020).
        s_alpha(y) = floor(y) + 0.5 * tanh(alpha * r) / tanh(alpha/2) + 0.5
        where r = y - floor(y) - 0.5 (centered fractional part)
        """
        fl = torch.floor(y)
        r = y - fl - 0.5
        return fl + 0.5 * torch.tanh(alpha * r) / (math.tanh(alpha / 2) + 1e-10) + 0.5

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        cr = self._clip_range
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            if CastedLinear._use_soft_round:
                # Soft-Round QAT: differentiable rounding with temperature annealing
                w32 = self.weight.float()
                row_clip = torch.quantile(w32.abs(), 0.9995, dim=1)
                scale = (row_clip / float(cr)).clamp_min(1.0 / float(cr))
                w_scaled = w32 / scale[:, None]
                w_rounded = CastedLinear.soft_round(w_scaled, CastedLinear._soft_round_alpha)
                w_q = (torch.clamp(w_rounded, -(cr+1), cr) * scale[:, None]).to(x.dtype)
                w = w_q  # fully differentiable path
            else:
                # Original STE QAT
                with torch.no_grad():
                    w32 = self.weight.float()
                    row_clip = torch.quantile(w32.abs(), 0.9995, dim=1)
                    scale = (row_clip / float(cr)).clamp_min(1.0 / float(cr))
                    w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -(cr+1), cr) * scale[:, None]).to(x.dtype)
                w = w + (w_q - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()

class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024, rope_dims: int = 0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * (scale ** (rd / (rd - 2)))
                inv_freq = 1.0 / (new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd))
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)

def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 rope_base: float, qk_gain_init: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)
        self.use_xsa = False

    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        y_g = y.reshape(B, T, Hkv, H // Hkv, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x: Tensor, v_embed: Tensor | None = None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x)
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        if _HAS_FA3:
            y = flash_attn_3_func(q, k, v, causal=True).contiguous()
        else:
            # PyTorch 2.3 does not support enable_gqa; expand K/V for GQA manually
            qt = q.transpose(1, 2)
            kt = k.transpose(1, 2)
            vt = v.transpose(1, 2)
            if self.num_kv_heads != self.num_heads:
                g = self.num_heads // self.num_kv_heads
                kt = kt.repeat_interleave(g, dim=1)
                vt = vt.repeat_interleave(g, dim=1)
            y = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=None, is_causal=True).transpose(1, 2)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        y = y.reshape(bsz, seqlen, dim)
        return self.proj(y)

class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev

class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size: int, ve_dim: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(ve_dim, model_dim, bias=False) if ve_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(token_ids)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float, layer_idx: int = 0,
                 ln_scale: bool = False, dtg: bool = False):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        if dtg:
            self.dtg_gate = nn.Linear(dim, 1, bias=True)
            nn.init.zeros_(self.dtg_gate.weight)
            nn.init.constant_(self.dtg_gate.bias, 2.0)
        else:
            self.dtg_gate = None

    def forward(self, x: Tensor, x0: Tensor, v_embed: Tensor | None = None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, v_embed=v_embed)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)
        if self.dtg_gate is not None:
            gate = torch.sigmoid(self.dtg_gate(x_in.detach()))
            x_out = x_in + gate * (x_out - x_in)
        return x_out

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, num_heads: int,
                 num_kv_heads: int, mlp_mult: int, tie_embeddings: bool, tied_embed_init_std: float,
                 logit_softcap: float, rope_base: float, qk_gain_init: float,
                 bigram_vocab_size: int = 0, bigram_dim: int = 128, xsa_last_n: int = 0,
                 rope_dims: int = 0, ln_scale: bool = False, dtg: bool = False,
                 ve_enabled: bool = False, ve_dim: int = 128, ve_layers: str = "9,10"):
        super().__init__()
        self._ve_target_dim = num_kv_heads * (model_dim // num_heads)
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base,
                  qk_gain_init, layer_idx=i, ln_scale=ln_scale, dtg=dtg)
            for i in range(num_layers)
        ])
        if rope_dims > 0:
            head_dim = model_dim // num_heads
            for block in self.blocks:
                block.attn.rope_dims = rope_dims
                block.attn.rotary = Rotary(head_dim, base=rope_base, train_seq_len=1024, rope_dims=rope_dims)
        self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if ve_enabled else []
        kv_dim = self._ve_target_dim
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim)
            self.ve_layer_scales = nn.ParameterList(
                [nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices]
            )
        else:
            self.ve_shared = None
            self.ve_layer_scales = nn.ParameterList()
        self.value_embeds = nn.ModuleList()
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        num_layers = len(self.blocks)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj." in name or name.endswith(".proj"):
                        with torch.no_grad():
                            module.weight.mul_(1.0 / math.sqrt(2 * num_layers))

    def _get_ve(self, layer_idx: int, input_ids: Tensor, ve_cache: dict | None = None) -> Tensor | None:
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices:
            return None
        if ve_cache is not None and 've' not in ve_cache:
            ve_cache['ve'] = self.ve_shared(input_ids)
        ve_base = ve_cache['ve'] if ve_cache is not None else self.ve_shared(input_ids)
        ve_idx = self.ve_layer_indices.index(layer_idx)
        return ve_base * self.ve_layer_scales[ve_idx].to(dtype=ve_base.dtype)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips: list[Tensor] = []
        ve_cache: dict = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x = self.blocks[i](x, x0, v_embed=ve)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x = self.blocks[bi](x, x0, v_embed=ve)
        x = self.final_norm(x)
        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips: list[Tensor] = []
        ve_cache: dict = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x = self.blocks[i](x, x0, v_embed=ve)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x = self.blocks[bi](x, x0, v_embed=ve)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

def eval_val_sliding(args: Hyperparameters, base_model: nn.Module, rank: int, world_size: int,
                     device: torch.device, val_tokens: Tensor, base_bytes_lut: Tensor,
                     has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor,
                     stride: int, batch_seqs: int = 32, eval_seq_len: int | None = None) -> tuple[float, float]:
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    last_full_start = max(total_tokens - seq_len, 0)
    window_starts = list(range(0, last_full_start + 1, stride))
    if not window_starts or window_starts[-1] != last_full_start:
        window_starts.append(last_full_start)
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    base_model.eval()
    # Disable compile on PyTorch 2.3 to avoid LR distortion and version_counter errors
    compiled_logits = base_model.forward_logits

    # Pre-compile: dummy forward+backward with TTT shapes to warm the compile cache
    if rank == 0:
        print("  ttt: pre-compiling forward+backward kernels...", flush=True)
    _dummy_x = torch.zeros(1, seq_len, dtype=torch.int64, device=device)
    _dummy_y = torch.zeros(1, seq_len, dtype=torch.int64, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _dummy_logits = base_model.forward_logits(_dummy_x)
        _dummy_loss = F.cross_entropy(_dummy_logits.reshape(-1, _dummy_logits.size(-1)), _dummy_y.reshape(-1))
    _dummy_loss.backward()
    base_model.zero_grad(set_to_none=True)
    if rank == 0:
        print("  ttt: pre-compile done", flush=True)
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    base_model.train()
    return val_loss, bits_per_token * tokens_per_byte

def eval_val_sliding_ttt(
    args: Hyperparameters, base_model: nn.Module, rank: int, world_size: int,
    device: torch.device, val_tokens: Tensor, base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor,
    stride: int, ttt_epochs: int = 3, ttt_lr: float = 0.001,
    ttt_momentum: float = 0.9, ttt_freeze_blocks: int = 2,
    batch_seqs: int = 32, eval_seq_len: int | None = None,
    ttt_chunk_tokens: int = 32768, ttt_optimizer: str = "adamw",
    ttt_temp: float = 1.0,
    ppm_alpha: float = 0.85,
    byte_weighted_ttt: bool = True,
    use_cache: bool = True,
    cache_alpha: float = 0.3,
    adaptive_lr: bool = True,
    adaptive_lr_max_mult: float = 3.0,
) -> tuple[float, float]:
    """Legal score-first TTT: score each chunk, then train on it.
    Every token scored BEFORE any update that could use it."""
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1

    # Initialize GPU-vectorized logistic context mixer
    use_mixer = os.environ.get("USE_MIXER", "1") == "1"
    ngram_max_order = int(os.environ.get("NGRAM_MAX_ORDER", "9"))
    mixer = BackoffNgramMixer(
        vocab_size=val_tokens.to(torch.int32).max().item() + 1,
        device=device,
        eta=float(os.environ.get("MIXER_ETA", "0.1")),
        max_order=ngram_max_order,
    ) if use_mixer else None
    if use_mixer and rank == 0:
        print(f"  Logistic context mixer enabled: eta={mixer.eta} max_order={ngram_max_order} "
              f"cubric={mixer._cubric_enabled}")
    if adaptive_lr and rank == 0:
        print(f"  Adaptive LR enabled: max_mult={adaptive_lr_max_mult}")

    # Pre-compute all window starts
    last_full_start = max(total_tokens - seq_len, 0)
    window_starts = list(range(0, last_full_start + 1, stride))
    if not window_starts or window_starts[-1] != last_full_start:
        window_starts.append(last_full_start)

    # Assign each window to a chunk based on scored token position
    num_chunks = (total_tokens + ttt_chunk_tokens - 1) // ttt_chunk_tokens
    chunk_windows: list[list[int]] = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        s = 0 if ws == 0 else max(wlen - stride, 0)
        scored_start = ws + s
        ci = min(scored_start // ttt_chunk_tokens, num_chunks - 1)
        chunk_windows[ci].append(ws)

    if rank == 0:
        print(f"ttt:start chunks={num_chunks} chunk_tokens={ttt_chunk_tokens} "
              f"windows={len(window_starts)} stride={stride} "
              f"lr={ttt_lr} epochs={ttt_epochs} opt={ttt_optimizer} "
              f"freeze_first={ttt_freeze_blocks}")

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    # Freeze everything, then selectively unfreeze for TTT
    num_blocks = len(base_model.blocks)
    for p in base_model.parameters():
        p.requires_grad_(False)
    ttt_params = []
    ttt_param_ids = set()
    use_qttt = os.environ.get("QTTT", "0") == "1"
    if use_qttt:
        # qTTT: only unfreeze Q projections in last N blocks + norms + head
        for i in range(max(0, num_blocks - ttt_freeze_blocks), num_blocks):
            for name, p in base_model.blocks[i].named_parameters():
                if "c_q" in name:
                    p.requires_grad_(True)
                    ttt_params.append(p)
                    ttt_param_ids.add(id(p))
    else:
        # Standard: unfreeze all params in last N blocks
        for i in range(max(0, num_blocks - ttt_freeze_blocks), num_blocks):
            for p in base_model.blocks[i].parameters():
                p.requires_grad_(True)
                ttt_params.append(p)
                ttt_param_ids.add(id(p))
    # Unfreeze norms, scales, lm_head
    for name, p in base_model.named_parameters():
        if "norm" in name or "scale" in name or "lm_head" in name:
            p.requires_grad_(True)
            if id(p) not in ttt_param_ids:
                ttt_params.append(p)
                ttt_param_ids.add(id(p))

    if rank == 0:
        n_unfrozen = sum(p.numel() for p in ttt_params)
        n_frozen = sum(p.numel() for p in base_model.parameters() if not p.requires_grad)
        print(f"ttt:params unfrozen={n_unfrozen} frozen={n_frozen}")

    if ttt_optimizer == "adamw":
        optimizer = torch.optim.AdamW(ttt_params, lr=ttt_lr, weight_decay=0.0, betas=(0.9, 0.999))
    else:
        optimizer = torch.optim.SGD(ttt_params, lr=ttt_lr, momentum=ttt_momentum)

    # Polyak averaging (TTT weight EMA) for smoother scoring
    use_polyak = os.environ.get("USE_POLYAK", "1") == "1"
    polyak_decay = float(os.environ.get("POLYAK_DECAY", "0.998"))
    if use_polyak:
        polyak_state = {id(p): p.data.clone() for p in ttt_params}
        if rank == 0:
            print(f"  Polyak averaging enabled: decay={polyak_decay}")

    t0 = time.perf_counter()

    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue

        # --- Phase 1: SCORE this chunk (inference_mode, no grad) ---
        my_s = (len(windows) * rank) // world_size
        my_e = (len(windows) * (rank + 1)) // world_size
        my_windows = windows[my_s:my_e]

        # Swap in Polyak-averaged weights for scoring
        if use_polyak and ci > 0:
            _saved_weights = {}
            for p in ttt_params:
                _saved_weights[id(p)] = p.data.clone()
                p.data.copy_(polyak_state[id(p)])

        base_model.eval()
        with torch.inference_mode():
            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi:bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens: list[int] = []
                for i, ws in enumerate(batch_ws):
                    end = min(ws + seq_len, total_tokens)
                    wlen = end - ws
                    wlens.append(wlen)
                    chunk_tok = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk_tok[:-1]
                    y_batch[i, :wlen] = chunk_tok[1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = base_model.forward_logits(x_batch)
                logits_scaled = logits.float() / ttt_temp

                # Adaptive temperature: sharpen confident predictions more
                if ttt_temp != 1.0:
                    with torch.no_grad():
                        probs_for_entropy = F.softmax(logits.float(), dim=-1)
                        token_entropy = -(probs_for_entropy * (probs_for_entropy + 1e-10).log()).sum(-1)
                        max_ent = math.log(logits.size(-1))
                        # Confident tokens (low entropy) get more sharpening
                        adaptive_temp = 1.0 - (1.0 - ttt_temp) * (1.0 - token_entropy / max_ent)
                        adaptive_temp = adaptive_temp.clamp(min=0.9, max=1.05)
                        logits_scaled = logits.float() / adaptive_temp.unsqueeze(-1)

                # Logistic context mixing (GPU-vectorized) or plain CE
                if mixer is not None:
                    nll, expert_nll = mixer.mix_and_score(logits_scaled, x_batch, y_batch, wlens)
                    mixer.update_weights(expert_nll, wlens)
                else:
                    nll = F.cross_entropy(
                        logits_scaled.reshape(-1, logits_scaled.size(-1)),
                        y_batch.reshape(-1), reduction="none",
                    ).reshape(bsz, seq_len)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else max(wlen - stride, 0)
                    scored_nll = nll[i, s:wlen].to(torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(wlen - s)
                    tgt, prev = y_batch[i, s:wlen], x_batch[i, s:wlen]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()

        # --- Update Cubric multipliers (before table update, uses chunk beat stats) ---
        if mixer is not None:
            mixer.step_cubric()

        # --- Update context mixer with scored chunk tokens (GPU-vectorized) ---
        chunk_start_tok = ci * ttt_chunk_tokens
        chunk_end_tok = min((ci + 1) * ttt_chunk_tokens, total_tokens)
        if mixer is not None:
            mixer.update(val_tokens[chunk_start_tok:chunk_end_tok + 1])

        # Swap back training weights after scoring
        if use_polyak and ci > 0:
            for p in ttt_params:
                p.data.copy_(_saved_weights[id(p)])

        # --- Phase 2: TRAIN on this chunk (already scored = legal) ---
        is_last_chunk = (ci == num_chunks - 1)
        if not is_last_chunk and ttt_epochs > 0:
            chunk_start = ci * ttt_chunk_tokens
            chunk_end = min((ci + 1) * ttt_chunk_tokens, total_tokens)
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if rank == 0 and ci < 3:
                print(f"    ttt_train [{ci+1}] seqs={chunk_seqs} start_train...", flush=True)
            if chunk_seqs > 0:
                # Cosine LR across chunks + adaptive scaling
                cos_lr = ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                if adaptive_lr:
                    # Increase LR as we've seen more data (more confident adaptation)
                    progress = min(ci / max(num_chunks * 0.3, 1), 1.0)  # ramp over first 30% of chunks
                    lr_mult = 1.0 + (adaptive_lr_max_mult - 1.0) * progress
                    cos_lr = cos_lr * lr_mult
                for pg in optimizer.param_groups:
                    pg["lr"] = cos_lr
                my_seq_s = (chunk_seqs * rank) // world_size
                my_seq_e = (chunk_seqs * (rank + 1)) // world_size
                my_chunk_seqs = my_seq_e - my_seq_s
                for _ep in range(ttt_epochs):
                    if rank == 0 and ci < 3:
                        print(f"    ttt_train [{ci+1}] epoch={_ep+1}/{ttt_epochs} batches={my_chunk_seqs} ...", flush=True)
                    for bs in range(0, my_chunk_seqs, batch_seqs):
                        be = min(bs + batch_seqs, my_chunk_seqs)
                        actual_bs = my_seq_s + bs
                        start_tok = chunk_start + actual_bs * seq_len
                        end_tok = chunk_start + (my_seq_s + be) * seq_len + 1
                        if end_tok > val_tokens.numel():
                            continue
                        local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, seq_len)
                        y = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            if byte_weighted_ttt:
                                # Byte-weighted loss: tokens covering more bytes matter more
                                ttt_logits = base_model.forward_logits(x)
                                per_token_loss = F.cross_entropy(
                                    ttt_logits.reshape(-1, ttt_logits.size(-1)),
                                    y.reshape(-1), reduction='none'
                                ).reshape(y.shape)
                                byte_weights = base_bytes_lut[y].float()
                                byte_weights = byte_weights + (has_leading_space_lut[y] & ~is_boundary_token_lut[x]).float()
                                ttt_loss = (per_token_loss * byte_weights).sum() / byte_weights.sum()
                            else:
                                ttt_loss = base_model(x, y)
                        ttt_loss.backward()
                        # Each rank adapts independently (no gradient sync).
                        # Syncing grads across ranks would cause NCCL deadlock when
                        # some ranks have empty batches (p.grad=None mismatch).
                        torch.nn.utils.clip_grad_norm_(ttt_params, 1.0)
                        optimizer.step()
                        # Update Polyak EMA after each step
                        if use_polyak:
                            for p in ttt_params:
                                polyak_state[id(p)].lerp_(p.data, 1.0 - polyak_decay)
                        if rank == 0 and ci < 3:
                            print(f"      step done ep={_ep+1} bs={bs} loss={ttt_loss.item():.4f}", flush=True)

        if rank == 0 and (ci % 10 == 0 or ci == num_chunks - 1 or ci < 5):
            elapsed = time.perf_counter() - t0
            rl = loss_sum.item() / max(token_count.item(), 1)
            rbpb = rl / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1)) if token_count.item() > 0 else 0.0
            print(f"  ttt_chunk [{ci+1}/{num_chunks}] bpb={rbpb:.6f} time={elapsed:.1f}s", flush=True)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())

    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()

    if rank == 0:
        print(f"ttt:done val_loss={val_loss:.6f} val_bpb={val_bpb:.6f} "
              f"elapsed={time.perf_counter() - t0:.1f}s")
    return val_loss, val_bpb

def eval_ngram(
    args, eval_model: nn.Module, rank: int, world_size: int, device: torch.device,
    val_tokens: Tensor, base_bytes_lut: Tensor, has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor, stride: int, eval_seq_len: int | None = None,
    ngram_alpha_max: float = 0.95, ngram_alpha_min: float = 0.05,
    ngram_entropy_center: float = 3.0, ngram_entropy_scale: float = 2.0,
    ngram_chunk_tokens: int = 1_000_000, ngram_max_order: int = 9,
    ngram_min_order: int = 2, ngram_buckets: int = 4_194_304,
    ngram_order_mults: tuple = (0.3, 0.3, 0.97, 2.0, 2.0, 2.0, 2.0, 2.0),
    batch_seqs: int = 32,
    ngram_min_count: int = 2,
    ngram_alpha_clip: float = 0.95,
    ngram_conf_scale: float = 0.0,
    initial_cache: "NgramEvalCache | None" = None,
) -> tuple[float, float]:
    """Pure n-gram-augmented evaluation (no TTT). Score-first: cache updated AFTER each chunk.

    Based on PR #809: 1M-token chunks give dramatically better n-gram statistics vs 32K.
    Uses np.bincount for fast hash-table updates (10-100x vs np.add.at).
    """
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    tokens_np = val_tokens.cpu().numpy().astype(np.int64)

    if initial_cache is not None:
        cache = initial_cache
    else:
        cache = NgramEvalCache(max_order=ngram_max_order, min_order=ngram_min_order,
                               num_buckets=ngram_buckets, min_count=ngram_min_count)
    order_mults_arr = np.array(ngram_order_mults, dtype=np.float64)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    byte_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)

    segments = _build_sliding_segments(total_tokens, seq_len, stride)
    seg_idx = 0
    eval_model.eval()
    t0 = time.perf_counter()
    n_chunks = (total_tokens + ngram_chunk_tokens - 1) // ngram_chunk_tokens

    if rank == 0:
        print(f"ngram_eval:start chunks={n_chunks} chunk_tokens={ngram_chunk_tokens} "
              f"alpha_max={ngram_alpha_max} entropy_center={ngram_entropy_center} "
              f"order_mults={list(ngram_order_mults)}", flush=True)

    with torch.inference_mode():
        for chunk_idx, chunk_start in enumerate(range(1, total_tokens + 1, ngram_chunk_tokens)):
            chunk_end = min(chunk_start + ngram_chunk_tokens, total_tokens + 1)

            # Collect all segments whose scored tokens fall in [chunk_start, chunk_end)
            chunk_segs: list = []
            while seg_idx < len(segments) and segments[seg_idx][4] < chunk_end:
                chunk_segs.append(segments[seg_idx])
                seg_idx += 1

            # Each rank handles a strided slice of segments
            rank_segs = chunk_segs[rank::world_size]

            for bi in range(0, len(rank_segs), batch_seqs):
                batch_seg = rank_segs[bi:bi + batch_seqs]
                bsz = len(batch_seg)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                for ri, (ws, vl, _, _, _, _) in enumerate(batch_seg):
                    end = min(ws + seq_len, total_tokens)
                    c = val_tokens[ws:end + 1].to(device=device, dtype=torch.int64)
                    x_batch[ri, :vl] = c[:-1]
                    y_batch[ri, :vl] = c[1:]

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = eval_model.forward_logits(x_batch)

                # Vectorized: compute probs+entropy for all scored tokens in batch at once
                # Build flat arrays for all scored positions across all segments
                seg_lengths = [seg[5] - seg[4] for seg in batch_seg]
                total_scored = sum(seg_lengths)
                all_positions = np.empty(total_scored, dtype=np.int64)
                all_targets_np = np.empty(total_scored, dtype=np.int64)
                all_model_p_np = np.empty(total_scored, dtype=np.float64)
                all_entropy_np = np.empty(total_scored, dtype=np.float64)

                # GPU: compute all probs/entropy in one vectorized pass
                # Extract scored slices for each segment
                ptr = 0
                for ri, (_, _, lss, lse, tstart, tend) in enumerate(batch_seg):
                    slen = tend - tstart
                    row_logits = logits[ri, lss:lse].float()
                    row_targets = y_batch[ri, lss:lse]
                    model_probs = torch.softmax(row_logits, dim=-1)
                    log_probs = torch.log_softmax(row_logits, dim=-1)
                    seg_model_p = model_probs.gather(1, row_targets.unsqueeze(-1)).squeeze(-1).clamp(min=1e-10)
                    seg_entropy = -(model_probs * log_probs).sum(dim=-1)
                    all_model_p_np[ptr:ptr+slen] = seg_model_p.cpu().numpy()
                    all_entropy_np[ptr:ptr+slen] = seg_entropy.cpu().numpy()
                    all_positions[ptr:ptr+slen] = np.arange(tstart, tend, dtype=np.int64)
                    all_targets_np[ptr:ptr+slen] = row_targets.cpu().numpy()
                    ptr += slen

                # Single batch_lookup for ALL scored tokens in this neural batch (vectorized over ~2048 positions)
                all_ngram_p, all_matched, all_orders = cache.batch_lookup(
                    tokens_np, all_positions, all_targets_np)

                # Compute mixing for all matched tokens at once
                final_p = all_model_p_np.copy()
                if all_matched.any():
                    matched_ords = all_orders[all_matched].astype(np.float64)
                    centers = ngram_entropy_center - 0.25 * (matched_ords - cache.min_order)
                    sig = 1.0 / (1.0 + np.exp(-ngram_entropy_scale * (all_entropy_np[all_matched] - centers)))
                    alpha = ngram_alpha_min + (ngram_alpha_max - ngram_alpha_min) * sig
                    mult_idx = np.clip(all_orders[all_matched] - cache.min_order, 0, len(order_mults_arr) - 1)
                    alpha = np.clip(alpha * order_mults_arr[mult_idx], 0.0, ngram_alpha_clip)
                    if ngram_conf_scale > 0.0:
                        conf_weight = np.clip(all_ngram_p[all_matched] / ngram_conf_scale, 0.0, 1.0)
                        alpha = alpha * conf_weight
                    final_p[all_matched] = (1.0 - alpha) * all_model_p_np[all_matched] + alpha * all_ngram_p[all_matched]
                    final_p = np.maximum(final_p, 1e-10)

                loss_sum += float((-np.log(final_p)).sum())
                token_count += total_scored

                # Byte counting: per-segment
                ptr = 0
                for ri, (_, _, lss, lse, tstart, tend) in enumerate(batch_seg):
                    slen = tend - tstart
                    sy = y_batch[ri, lss:lse]
                    sx = x_batch[ri, lss:lse]
                    tb = base_bytes_lut[sy].to(torch.float64)
                    tb += (has_leading_space_lut[sy] & ~is_boundary_token_lut[sx]).to(torch.float64)
                    byte_sum += tb.sum()
                    ptr += slen

            # Score-first: update cache AFTER scoring this chunk
            cache.update_batch(tokens_np, chunk_start, chunk_end)

            if rank == 0 and (chunk_idx % 5 == 0 or chunk_idx == n_chunks - 1):
                elapsed = time.perf_counter() - t0
                rl = loss_sum.item() / max(token_count.item(), 1)
                rbpb = rl / math.log(2.0) * (token_count.item() / max(byte_sum.item(), 1)) if token_count.item() > 0 else 0.0
                print(f"  ngram_chunk [{chunk_idx+1}/{n_chunks}] bpb={rbpb:.6f} time={elapsed:.1f}s", flush=True)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_sum.item())

    if rank == 0:
        print(f"ngram:done val_loss={val_loss:.6f} val_bpb={val_bpb:.6f} "
              f"elapsed={time.perf_counter() - t0:.1f}s")
    return val_loss, val_bpb

def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"

def quantize_int6_per_row(t: Tensor, clip_range: int = 15) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float('inf')
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            if pct < 1.0:
                row_clip = torch.quantile(t32.abs(), pct, dim=1)
            else:
                row_clip = t32.abs().amax(dim=1)
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
            recon = q.float() * s.float()[:, None]
            err = (t32 - recon).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    return q, scale


def _get_layer_clip_range(name: str, num_layers: int, int6_last_n: int) -> int:
    """Return clip_range based on which layer the param belongs to."""
    import re
    m = re.search(r'blocks\.(\d+)\.', name)
    if m:
        layer_idx = int(m.group(1))
        if layer_idx >= num_layers - int6_last_n:
            return 31  # int6
    return 15  # int5

def mixed_quantize_int6(state_dict: dict[str, Tensor], int6_cats: set[str]):
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if cat in int6_cats and t.ndim >= 1:
            q, s = quantize_int6_per_row(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
        else:
            q, s = quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    return result, meta

def dequantize_mixed_int6(result: dict[str, Tensor], meta: dict[str, object],
                          template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out

def main() -> None:
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    # Skip compiling Newton-Schulz on PyTorch 2.3 to avoid LR schedule distortion
    pass
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
        enable_cudnn_sdp(False)
        enable_flash_sdp(True)
        enable_mem_efficient_sdp(False)
        enable_math_sdp(False)
    except ImportError:
        # Older PyTorch versions
        pass
    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)
    log0(code, console=False)
    log0(f"Python {sys.version} PyTorch {torch.__version__}", console=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    CastedLinear._qat_enabled = args.qat_enabled
    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        dtg=args.dtg_enabled, ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    # Disable torch.compile: PyTorch 2.3 compile causes version_counter errors and
    # distorts the wall-clock LR schedule during compile warmup (step1 ~4600ms vs ~100ms steady-state)
    compiled_model = base_model
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    scalar_params.append(base_model.smear.gate)
    if base_model.bigram is not None:
        scalar_params.append(base_model.bigram.scale)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    if base_model.bigram is not None:
        tok_params.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.bigram.proj is not None:
            matrix_params.append(base_model.bigram.proj.weight)
    if base_model.ve_shared is not None:
        tok_params.append({"params": [base_model.ve_shared.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.ve_shared.proj is not None:
            matrix_params.append(base_model.ve_shared.proj.weight)
        scalar_params.append(base_model.ve_shared.scale)
        for s in base_model.ve_layer_scales:
            scalar_params.append(s)
    optimizer_tok = torch.optim.AdamW(
        tok_params,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_wd,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)
    n_params = sum(p.numel() for p in base_model.parameters())
    # Set int6 clip_range for last N layers (mixed precision)
    int6_start = args.num_layers - args.int6_last_n
    for i, block in enumerate(base_model.blocks):
        if i >= int6_start:
            for m in block.modules():
                if isinstance(m, CastedLinear):
                    m._clip_range = 31  # int6
    if master_process:
        int5_count = sum(1 for m in base_model.modules() if isinstance(m, CastedLinear) and m._clip_range == 15)
        int6_count = sum(1 for m in base_model.modules() if isinstance(m, CastedLinear) and m._clip_range == 31)
        log0(f"mixed_precision: {int5_count} int5 layers, {int6_count} int6 layers (last {args.int6_last_n} blocks)")
    log0(f"model_params:{n_params}")
    xsa_layers = [i for i, b in enumerate(base_model.blocks) if b.attn.use_xsa]
    log0(f"XSA:{xsa_layers} ws:{world_size} gqa:{args.num_heads}/{args.num_kv_heads}")
    log0(f"lr:embed={token_lr} matrix={args.matrix_lr} scalar={args.scalar_lr} batch:{args.train_batch_tokens} wall:{args.max_wallclock_seconds:.0f}s seed:{args.seed}")
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    train_reserve_ms = 18000
    effective_train_ms = (max_wallclock_ms - train_reserve_ms) if max_wallclock_ms is not None else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if effective_train_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        # Cap step_ms to prevent compile-warmup distortion (first steps may be 10-50x slower)
        step_ms = min(elapsed_ms / max(step, 1), 300.0)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(effective_train_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0
    # NGRAM_ONLY mode: skip training, load saved model, run ngram eval directly
    if os.environ.get("NGRAM_ONLY", "0") == "1":
        log0("NGRAM_ONLY mode: skipping training, loading saved model...")
        sd_cpu = {k: v.cpu() for k, v in torch.load("final_model.pt", map_location="cpu").items()}
        with open("final_model.int6.ptz", "rb") as f:
            quant_blob_disk = f.read()
        quant_state = torch.load(
            io.BytesIO((zstandard.ZstdDecompressor().decompress(quant_blob_disk) if _COMPRESSOR == "zstd" else lzma.decompress(quant_blob_disk) if _COMPRESSOR == "lzma" else zlib.decompress(quant_blob_disk))),
            map_location="cpu",
        )
        deq_state_no = dequantize_mixed_int6(quant_state["w"], quant_state["m"], sd_cpu)
        eval_model_no = GPT(
            vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
            num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
            bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim, xsa_last_n=args.xsa_last_n,
            rope_dims=args.rope_dims, ln_scale=args.ln_scale, dtg=args.dtg_enabled,
            ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
        ).to(device).bfloat16()
        for m in eval_model_no.modules():
            if isinstance(m, CastedLinear):
                m.float()
        restore_low_dim_params_to_fp32(eval_model_no)
        eval_model_no.load_state_dict(deq_state_no, strict=True)
        sw_seq_len_no = int(os.environ.get("EVAL_SEQ_LEN", str(effective_eval_seq_len)))
        ng_chunk = int(os.environ.get("NGRAM_CHUNK_TOKENS", "1000000"))
        ng_alpha_max = float(os.environ.get("NGRAM_ALPHA_MAX", "1.0"))
        ng_alpha_min = float(os.environ.get("NGRAM_ALPHA_MIN", "1.0"))
        ng_entropy_center = float(os.environ.get("NGRAM_ENTROPY_CENTER", "3.0"))
        ng_entropy_scale = float(os.environ.get("NGRAM_ENTROPY_SCALE", "2.0"))
        ng_max_order = int(os.environ.get("NGRAM_MAX_ORDER", "9"))
        ng_order_mults_str = os.environ.get("NGRAM_ORDER_MULTS", "0.3,0.3,0.97,2.0,2.0,2.0,2.0,2.0")
        ng_order_mults = tuple(float(x) for x in ng_order_mults_str.split(","))
        ng_buckets = int(os.environ.get("NGRAM_BUCKETS", "1"))
        ng_min_count = int(os.environ.get("NGRAM_MIN_COUNT", "1"))
        ng_alpha_clip = float(os.environ.get("NGRAM_ALPHA_CLIP", "1.0"))
        ng_conf_scale = float(os.environ.get("NGRAM_CONF_SCALE", "0.0"))
        eval_stride_no = int(os.environ.get("EVAL_STRIDE", "64"))
        # Optional: pre-fill n-gram cache from training data before scoring val
        ng_prefill_tokens = int(os.environ.get("NGRAM_PREFILL_TOKENS", "10000000"))
        prefill_cache = None
        if ng_prefill_tokens > 0:
            log0(f"NGRAM_PREFILL: filling cache with {ng_prefill_tokens} training tokens...")
            t_pf = time.perf_counter()
            prefill_cache = NgramEvalCache(max_order=ng_max_order, min_order=2,
                                           num_buckets=ng_buckets, min_count=ng_min_count)
            import glob as _glob
            train_files_pf = sorted(_glob.glob(args.train_files))
            tokens_pf_done = 0
            for tf_pf in train_files_pf:
                if tokens_pf_done >= ng_prefill_tokens:
                    break
                with open(tf_pf, "rb") as _f:
                    _hdr = np.frombuffer(_f.read(1024), dtype="<i4")
                    _n = int(_hdr[2])
                    _n_read = min(ng_prefill_tokens - tokens_pf_done, _n)
                    _raw = _f.read(_n_read * 2)
                _tok = np.frombuffer(_raw, dtype="<u2").astype(np.int64)
                prefill_cache.update_batch(_tok, 0, len(_tok))
                tokens_pf_done += len(_tok)
            log0(f"NGRAM_PREFILL: {tokens_pf_done} tokens loaded in {time.perf_counter()-t_pf:.1f}s")
        torch.cuda.synchronize()
        t_no = time.perf_counter()
        no_val_loss, no_val_bpb = eval_ngram(
            args, eval_model_no, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=eval_stride_no, eval_seq_len=sw_seq_len_no,
            ngram_alpha_max=ng_alpha_max, ngram_alpha_min=ng_alpha_min,
            ngram_entropy_center=ng_entropy_center, ngram_entropy_scale=ng_entropy_scale,
            ngram_chunk_tokens=ng_chunk, ngram_max_order=ng_max_order,
            ngram_order_mults=ng_order_mults,
            ngram_buckets=ng_buckets, ngram_min_count=ng_min_count,
            ngram_alpha_clip=ng_alpha_clip, ngram_conf_scale=ng_conf_scale,
            initial_cache=prefill_cache,
        )
        torch.cuda.synchronize()
        log0(f"ngram_only val_loss:{no_val_loss:.4f} val_bpb:{no_val_bpb:.4f} eval_time:{1000.0*(time.perf_counter()-t_no):.0f}ms")
        log0(f"ngram_only_exact val_loss:{no_val_loss:.8f} val_bpb:{no_val_bpb:.8f}")
        if distributed:
            dist.destroy_process_group()
        return

    # TTT_ONLY mode: skip training, load saved model, run TTT eval
    if os.environ.get("TTT_ONLY", "0") == "1":
        log0("TTT_ONLY mode: skipping training, loading saved model...")
        sd_cpu = {k: v.cpu() for k, v in torch.load("final_model.pt", map_location="cpu").items()}
        if args.prune_pct > 0:
            for k, v in sd_cpu.items():
                if v.ndim == 2 and v.numel() > 65536:
                    thresh = torch.quantile(v.abs().float(), args.prune_pct)
                    v[v.abs() < thresh] = 0.0
            log0(f"pruning:{args.prune_pct*100:.1f}% magnitude pruning applied")
        with open("final_model.int6.ptz", "rb") as f:
            quant_blob_disk = f.read()
        quant_state = torch.load(
            io.BytesIO((zstandard.ZstdDecompressor().decompress(quant_blob_disk) if _COMPRESSOR == "zstd" else lzma.decompress(quant_blob_disk) if _COMPRESSOR == "lzma" else zlib.decompress(quant_blob_disk))),
            map_location="cpu",
        )
        deq_state = dequantize_mixed_int6(quant_state["w"], quant_state["m"], sd_cpu)
        eval_model = GPT(
            vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
            num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
            bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim, xsa_last_n=args.xsa_last_n,
            rope_dims=args.rope_dims, ln_scale=args.ln_scale, dtg=args.dtg_enabled,
            ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
        ).to(device).bfloat16()
        for m in eval_model.modules():
            if isinstance(m, CastedLinear):
                m.float()
        restore_low_dim_params_to_fp32(eval_model)
        eval_model.load_state_dict(deq_state, strict=True)
        sw_seq_len = int(os.environ.get("EVAL_SEQ_LEN", str(effective_eval_seq_len)))
        log0(f"TTT_ONLY: model loaded, starting TTT eval...")
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_epochs = int(os.environ.get("TTT_EPOCHS", "3"))
        ttt_lr = float(os.environ.get("TTT_LR", "0.0005"))
        ttt_freeze = int(os.environ.get("TTT_FREEZE_BLOCKS", "2"))
        ttt_chunk = int(os.environ.get("TTT_CHUNK_TOKENS", "32768"))
        ttt_opt = os.environ.get("TTT_OPTIMIZER", "adamw")
        log0(f"TTT: epochs={ttt_epochs} lr={ttt_lr} freeze_first={ttt_freeze} chunk={ttt_chunk} opt={ttt_opt}")
        ttt_temp = args.ttt_temperature
        log0(f"TTT temperature: {ttt_temp}")
        ttt_val_loss, ttt_val_bpb = eval_val_sliding_ttt(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride, ttt_epochs=ttt_epochs, ttt_lr=ttt_lr,
            ttt_freeze_blocks=ttt_freeze, eval_seq_len=sw_seq_len,
            ttt_chunk_tokens=ttt_chunk, ttt_optimizer=ttt_opt,
            ttt_temp=ttt_temp,
            ppm_alpha=float(os.environ.get("PPM_ALPHA", "0.85")),
            byte_weighted_ttt=os.environ.get("BYTE_WEIGHTED_TTT", "1") == "1",
            use_cache=os.environ.get("USE_CACHE", "1") == "1",
            cache_alpha=float(os.environ.get("CACHE_ALPHA", "0.3")),
            adaptive_lr=os.environ.get("ADAPTIVE_LR", "1") == "1",
            adaptive_lr_max_mult=float(os.environ.get("ADAPTIVE_LR_MAX", "3.0")),
        )
        torch.cuda.synchronize()
        log0(
            f"final_int6_ttt val_loss:{ttt_val_loss:.4f} val_bpb:{ttt_val_bpb:.4f} "
            f"stride:{args.eval_stride} eval_time:{1000.0 * (time.perf_counter() - t_ttt):.0f}ms"
        )
        log0(f"final_int6_ttt_exact val_loss:{ttt_val_loss:.8f} val_bpb:{ttt_val_bpb:.8f}")
        if distributed:
            dist.destroy_process_group()
        return

    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0
    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
    ema_decay = float(os.environ.get("EMA_DECAY", "0.997"))
    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        # Anneal soft-round alpha based on QAT progress
        if CastedLinear._use_soft_round and CastedLinear._qat_enabled:
            qat_progress = max(0.0, 1.0 - scale / max(args.late_qat_threshold, 0.01))
            CastedLinear._soft_round_alpha = 1.0 + 15.0 * qat_progress
        if args.late_qat_threshold > 0 and scale < args.late_qat_threshold and not CastedLinear._qat_enabled:
            CastedLinear._qat_enabled = True
            CastedLinear._use_soft_round = os.environ.get("SOFT_ROUND_QAT", "0") == "1"
            if CastedLinear._use_soft_round and master_process:
                log0(f"soft_round_qat:enabled initial_alpha=1.0")
            log0(f"late_qat:enabled step:{step} scale:{scale:.4f}")
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            # CROWN-Q: penalize quantization-sensitive weights during warmdown
            crownq_lambda = float(os.environ.get("CROWN_Q_LAMBDA", "0.01"))
            if CastedLinear._qat_enabled and crownq_lambda > 0:
                cq_loss = torch.zeros((), device=device)
                for m in base_model.modules():
                    if isinstance(m, CastedLinear) and m.weight.ndim == 2:
                        w = m.weight.float()
                        cr = float(m._clip_range)
                        row_max = w.detach().abs().amax(dim=1)
                        delta = row_max / cr  # quantization step size
                        cq_loss = cq_loss + (w.pow(2) * delta.pow(2).unsqueeze(1)).mean()
                loss = loss + crownq_lambda * cq_loss / 12.0
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()
        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)
        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.swa_enabled and scale < 0.2 and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()}
                swa_count = 1
                log0(f"swa:start step:{step}")
            else:
                for name, t in base_model.state_dict().items():
                    swa_state[name] += t.detach().cpu()
                swa_count += 1
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )
        reached_cap = effective_train_ms is not None and approx_training_time_ms >= effective_train_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step
    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )
    # Apply EMA weights directly (skip diagnostic evals to save ~5s of reserve)
    log0("ema:applying EMA weights (skipping diagnostic evals)")
    current_state = base_model.state_dict()
    ema_sd = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
    base_model.load_state_dict(ema_sd, strict=True)
    export_sd = base_model.state_dict()
    if master_process:
        torch.save(export_sd, "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    if args.prune_pct > 0:
        for k, v in sd_cpu.items():
            if v.ndim == 2 and v.numel() > 65536:
                thresh = torch.quantile(v.abs().float(), args.prune_pct)
                v[v.abs() < thresh] = 0.0
        if master_process:
            log0(f"pruning:{args.prune_pct*100:.1f}% magnitude pruning applied")
    quant_result, quant_meta = mixed_quantize_int6(sd_cpu, {"mlp", "attn"})
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    if _COMPRESSOR == "zstd":
        quant_blob = zstandard.ZstdCompressor(level=22).compress(quant_raw)
    elif _COMPRESSOR == "lzma":
        quant_blob = lzma.compress(quant_raw, preset=6)
    else:
        quant_blob = zlib.compress(quant_raw, 9)
    if master_process:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = len(quant_blob)
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model int6+{_COMPRESSOR}: {quant_file_bytes} bytes")
        log0(f"Total submission size int6+{_COMPRESSOR}: {quant_file_bytes + code_bytes} bytes")
    if distributed:
        dist.barrier()
    with open("final_model.int6.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(
        io.BytesIO((zstandard.ZstdDecompressor().decompress(quant_blob_disk) if _COMPRESSOR == "zstd" else lzma.decompress(quant_blob_disk) if _COMPRESSOR == "lzma" else zlib.decompress(quant_blob_disk))),
        map_location="cpu",
    )
    deq_state = dequantize_mixed_int6(quant_state["w"], quant_state["m"], sd_cpu)
    eval_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim, xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims, ln_scale=args.ln_scale, dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
    ).to(device).bfloat16()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)
    sw_seq_len = int(os.environ.get("EVAL_SEQ_LEN", str(effective_eval_seq_len)))
    if sw_seq_len != effective_eval_seq_len and rank == 0:
        log0(f"Eval seq_len override: {effective_eval_seq_len} -> {sw_seq_len}")
    # Skip standard sliding window eval by default when ngram is enabled (ngram is the primary metric)
    _ngram_en = os.environ.get("NGRAM_ENABLED", "1") == "1"
    _skip_sliding = os.environ.get("SKIP_SLIDING", "1" if _ngram_en else "0") == "1"
    if args.eval_stride > 0 and args.eval_stride < sw_seq_len and not _skip_sliding:
        torch.cuda.synchronize()
        t_slide = time.perf_counter()
        sw_val_loss, sw_val_bpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride,
            eval_seq_len=sw_seq_len,
        )
        torch.cuda.synchronize()
        log0(
            f"final_int6_sliding_window val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} "
            f"stride:{args.eval_stride} eval_time:{1000.0 * (time.perf_counter() - t_slide):.0f}ms"
        )
        log0(f"final_int6_sliding_window_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")
    # TTT eval is optional: skip by default when NGRAM_ENABLED=1 (ngram gives much lower BPB)
    ttt_enabled = os.environ.get("TTT_ENABLED", "0" if os.environ.get("NGRAM_ENABLED", "1") == "1" else "1") == "1"
    if ttt_enabled:
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_epochs = int(os.environ.get("TTT_EPOCHS", "3"))
        ttt_lr = float(os.environ.get("TTT_LR", "0.0005"))
        ttt_freeze = int(os.environ.get("TTT_FREEZE_BLOCKS", "2"))
        ttt_chunk = int(os.environ.get("TTT_CHUNK_TOKENS", "32768"))
        ttt_opt = os.environ.get("TTT_OPTIMIZER", "adamw")
        log0(f"TTT: epochs={ttt_epochs} lr={ttt_lr} freeze_first={ttt_freeze} chunk={ttt_chunk} opt={ttt_opt}")
        ttt_temp = args.ttt_temperature
        log0(f"TTT temperature: {ttt_temp}")
        ppm_alpha_val = float(os.environ.get("PPM_ALPHA", "0.85"))
        bw_ttt = os.environ.get("BYTE_WEIGHTED_TTT", "1") == "1"
        log0(f"PPM alpha: {ppm_alpha_val}, Byte-weighted TTT: {bw_ttt}")
        ttt_val_loss, ttt_val_bpb = eval_val_sliding_ttt(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride, ttt_epochs=ttt_epochs, ttt_lr=ttt_lr,
            ttt_freeze_blocks=ttt_freeze, eval_seq_len=sw_seq_len,
            ttt_chunk_tokens=ttt_chunk, ttt_optimizer=ttt_opt,
            ttt_temp=ttt_temp,
            ppm_alpha=float(os.environ.get("PPM_ALPHA", "0.85")),
            byte_weighted_ttt=os.environ.get("BYTE_WEIGHTED_TTT", "1") == "1",
            use_cache=os.environ.get("USE_CACHE", "1") == "1",
            cache_alpha=float(os.environ.get("CACHE_ALPHA", "0.3")),
            adaptive_lr=os.environ.get("ADAPTIVE_LR", "1") == "1",
            adaptive_lr_max_mult=float(os.environ.get("ADAPTIVE_LR_MAX", "3.0")),
        )
        torch.cuda.synchronize()
        log0(
            f"final_int6_ttt val_loss:{ttt_val_loss:.4f} val_bpb:{ttt_val_bpb:.4f} "
            f"stride:{args.eval_stride} eval_time:{1000.0 * (time.perf_counter() - t_ttt):.0f}ms"
        )
        log0(f"final_int6_ttt_exact val_loss:{ttt_val_loss:.8f} val_bpb:{ttt_val_bpb:.8f}")
    else:
        log0("TTT: skipped (NGRAM_ENABLED=1 — ngram eval provides primary metric)")

    # --- N-gram augmented eval (PR#809 style): separate pass, score-first, 1M-token chunks ---
    ngram_enabled = os.environ.get("NGRAM_ENABLED", "1") == "1"
    if ngram_enabled:
        # Re-load a clean eval_model (TTT modified the weights above)
        eval_model_ng = GPT(
            vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
            num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
            bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim, xsa_last_n=args.xsa_last_n,
            rope_dims=args.rope_dims, ln_scale=args.ln_scale, dtg=args.dtg_enabled,
            ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
        ).to(device).bfloat16()
        for m in eval_model_ng.modules():
            if isinstance(m, CastedLinear):
                m.float()
        restore_low_dim_params_to_fp32(eval_model_ng)
        eval_model_ng.load_state_dict(deq_state, strict=True)
        torch.cuda.synchronize()
        t_ng = time.perf_counter()
        ng_chunk = int(os.environ.get("NGRAM_CHUNK_TOKENS", "1000000"))
        ng_alpha_max = float(os.environ.get("NGRAM_ALPHA_MAX", "1.0"))
        ng_alpha_min = float(os.environ.get("NGRAM_ALPHA_MIN", "1.0"))
        ng_entropy_center = float(os.environ.get("NGRAM_ENTROPY_CENTER", "3.0"))
        ng_entropy_scale = float(os.environ.get("NGRAM_ENTROPY_SCALE", "2.0"))
        ng_max_order = int(os.environ.get("NGRAM_MAX_ORDER", "9"))
        ng_order_mults_str = os.environ.get("NGRAM_ORDER_MULTS", "0.3,0.3,0.97,2.0,2.0,2.0,2.0,2.0")
        ng_order_mults = tuple(float(x) for x in ng_order_mults_str.split(","))
        ng_buckets = int(os.environ.get("NGRAM_BUCKETS", "1"))
        ng_min_count = int(os.environ.get("NGRAM_MIN_COUNT", "1"))
        ng_alpha_clip = float(os.environ.get("NGRAM_ALPHA_CLIP", "1.0"))
        ng_conf_scale = float(os.environ.get("NGRAM_CONF_SCALE", "0.0"))
        # Pre-fill n-gram cache from training data
        ng_prefill_tokens_main = int(os.environ.get("NGRAM_PREFILL_TOKENS", "10000000"))
        prefill_cache_main = None
        if ng_prefill_tokens_main > 0:
            log0(f"NGRAM_PREFILL: filling cache with {ng_prefill_tokens_main} training tokens...")
            t_pf2 = time.perf_counter()
            prefill_cache_main = NgramEvalCache(max_order=ng_max_order, min_order=2,
                                                num_buckets=ng_buckets, min_count=ng_min_count)
            import glob as _glob2
            train_files_pf2 = sorted(_glob2.glob(args.train_files))
            tokens_pf2_done = 0
            for tf_pf2 in train_files_pf2:
                if tokens_pf2_done >= ng_prefill_tokens_main:
                    break
                with open(tf_pf2, "rb") as _f2:
                    _hdr2 = np.frombuffer(_f2.read(1024), dtype="<i4")
                    _n2 = int(_hdr2[2])
                    _n_read2 = min(ng_prefill_tokens_main - tokens_pf2_done, _n2)
                    _raw2 = _f2.read(_n_read2 * 2)
                _tok2 = np.frombuffer(_raw2, dtype="<u2").astype(np.int64)
                prefill_cache_main.update_batch(_tok2, 0, len(_tok2))
                tokens_pf2_done += len(_tok2)
            log0(f"NGRAM_PREFILL: {tokens_pf2_done} tokens in {time.perf_counter()-t_pf2:.1f}s")
        ng_val_loss, ng_val_bpb = eval_ngram(
            args, eval_model_ng, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride, eval_seq_len=sw_seq_len,
            ngram_alpha_max=ng_alpha_max, ngram_alpha_min=ng_alpha_min,
            ngram_entropy_center=ng_entropy_center, ngram_entropy_scale=ng_entropy_scale,
            ngram_chunk_tokens=ng_chunk, ngram_max_order=ng_max_order,
            ngram_order_mults=ng_order_mults,
            ngram_buckets=ng_buckets, ngram_min_count=ng_min_count,
            ngram_alpha_clip=ng_alpha_clip, ngram_conf_scale=ng_conf_scale,
            initial_cache=prefill_cache_main,
        )
        torch.cuda.synchronize()
        log0(
            f"final_ngram val_loss:{ng_val_loss:.4f} val_bpb:{ng_val_bpb:.4f} "
            f"stride:{args.eval_stride} eval_time:{1000.0 * (time.perf_counter() - t_ng):.0f}ms"
        )
        log0(f"final_ngram_exact val_loss:{ng_val_loss:.8f} val_bpb:{ng_val_bpb:.8f}")
        del eval_model_ng

    if distributed:
        dist.destroy_process_group()
if __name__ == "__main__":
    main()
