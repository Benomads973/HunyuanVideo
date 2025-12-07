import os
import math
import importlib.metadata

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    flash_attn = None
    flash_attn_varlen_func = None
    _flash_attn_forward = None


MEMORY_LAYOUT = {
    "flash": (
        # [b, s, h, d] → [(b*h), s, d]
        lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]),
        lambda x: x,
    ),
    "torch": (
        # [b, s, h, d] → [b, h, s, d]
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "vanilla": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
}


# =====================================================================
# Helpers
# =====================================================================

def _can_use_flash_attn() -> bool:
    """Retourne True si on PEUT utiliser flash-attn dans de bonnes conditions."""
    if flash_attn is None or flash_attn_varlen_func is None:
        return False
    if os.environ.get("HYVIDEO_DISABLE_FLASH_ATTN", "0") == "1":
        return False
    if not torch.cuda.is_available():
        return False

    # On force Ampere+ (sm_80 minimum) pour être safe
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        return False

    return True


def get_cu_seqlens(text_mask, img_len):
    """Calculate cu_seqlens_q, cu_seqlens_kv using text_mask and img_len

    Args:
        text_mask (torch.Tensor): the mask of text
        img_len (int): the length of image

    Returns:
        torch.Tensor: the calculated cu_seqlens for flash attention
    """
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len

    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device="cuda")

    for i in range(batch_size):
        s = text_len[i] + img_len
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    return cu_seqlens


# =====================================================================
# Core attention
# =====================================================================

def attention(
    q,
    k,
    v,
    mode="flash",
    drop_rate=0,
    attn_mask=None,
    causal=False,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    batch_size=1,
):
    """
    Perform QKV self attention.

    Args:
        q (torch.Tensor): [b, s, h, d]
        k (torch.Tensor): [b, s1, h, d]
        v (torch.Tensor): [b, s1, h, d]
        mode (str): 'flash', 'torch', 'vanilla'
        drop_rate (float): dropout rate on attention map
        attn_mask (torch.Tensor): see comment dans le code
        causal (bool): causal attention
        cu_seqlens_q / cu_seqlens_kv: pour flash-attn varlen
        max_seqlen_q / max_seqlen_kv: idem

    Returns:
        torch.Tensor: [b, s, h*d]
    """

    # -----------------------------------------------------------------
    # PATCH pour ton contexte :
    # - si mode == "flash" mais flash-attn inutilisable → fallback "torch"
    # -----------------------------------------------------------------
    if mode == "flash" and not _can_use_flash_attn():
        # Tu peux logger ici si tu veux voir le fallback :
        # print("[HunyuanVideo] FlashAttention indisponible → fallback sur 'torch'")
        mode = "torch"

    if mode not in MEMORY_LAYOUT:
        raise NotImplementedError(f"Unsupported attention mode: {mode}")

    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]
    q = pre_attn_layout(q)  # e.g. [b, s, h, d] → [b, h, s, d] ou (b*h, s, d)
    k = pre_attn_layout(k)
    v = pre_attn_layout(v)

    # -----------------------------------------------------------------
    # Mode PyTorch (scaled_dot_product_attention)
    # -----------------------------------------------------------------
    if mode == "torch":
        # q, k, v: [b, h, s, d]
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)

        if cu_seqlens_q is None:
            # Cas standard : pas de séquences varlen
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
            )
        else:
            # Cas Hunyuan : on split texte / image via les cu_seqlens
            attn1 = F.scaled_dot_product_attention(
                q[:, :, :cu_seqlens_q[1]],
                k[:, :, :cu_seqlens_kv[1]],
                v[:, :, :cu_seqlens_kv[1]],
                attn_mask=attn_mask,
                dropout_p=drop_rate,
                is_causal=causal,
            )
            attn2 = F.scaled_dot_product_attention(
                q[:, :, cu_seqlens_q[1] :],
                k[:, :, cu_seqlens_kv[1] :],
                v[:, :, cu_seqlens_kv[1] :],
                attn_mask=None,
                dropout_p=drop_rate,
                is_causal=False,
            )
            x = torch.cat([attn1, attn2], dim=2)

    # -----------------------------------------------------------------
    # Mode FlashAttention (varlen)
    # -----------------------------------------------------------------
    elif mode == "flash":
        # ici on sait que _can_use_flash_attn() == True
        if cu_seqlens_q is None or cu_seqlens_kv is None:
            raise ValueError("Flash mode requires cu_seqlens_q and cu_seqlens_kv")

        x = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        )
        # x: [(b*h*s), d] ou [(b*s), h, d] selon layout
        x = x.view(
            batch_size, max_seqlen_q, x.shape[-2], x.shape[-1]
        )  # [b, s, h, d]

    # -----------------------------------------------------------------
    # Mode vanilla (matmul + softmax)
    # -----------------------------------------------------------------
    elif mode == "vanilla":
        scale_factor = 1.0 / math.sqrt(q.size(-1))  # d^-0.5

        b, h, s, _ = q.shape
        s1 = k.size(2)
        attn_bias = torch.zeros(b, h, s, s1, dtype=q.dtype, device=q.device)

        if causal:
            # Causal uniquement pour self-attn
            assert (
                attn_mask is None
            ), "Causal mask and attn_mask cannot be used together"
            temp_mask = torch.ones(
                b, h, s, s, dtype=torch.bool, device=q.device
            ).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(q.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        attn = (q @ k.transpose(-2, -1)) * scale_factor
        attn += attn_bias
        attn = attn.softmax(dim=-1)
        attn = torch.dropout(attn, p=drop_rate, train=True)
        x = attn @ v

    else:
        raise NotImplementedError(f"Unsupported attention mode: {mode}")

    x = post_attn_layout(x)  # revenir en [b, s, h, d]
    b, s, h, d = x.shape
    out = x.reshape(b, s, -1)  # [b, s, h*d]
    return out


# =====================================================================
# parallel_attention (xfuser + flash_attn) – plutôt pour gros GPU
# =====================================================================

def parallel_attention(
    hybrid_seq_parallel_attn,
    q,
    k,
    v,
    img_q_len,
    img_kv_len,
    cu_seqlens_q,
    cu_seqlens_kv,
):
    """
    Version hybride pour les configs avec xfuser + flash-attn.
    Dans ton contexte (mono-GPU, pas Ampere), ce code ne devrait
    normalement PAS être appelé (ulysses_degree=ring_degree=1).
    """
    # Sécurité : si pas de flash_attn ou GPU inadapté → on fallback sur
    # la partie hybrid_seq_parallel_attn uniquement (sans second bloc).
    if not _can_use_flash_attn() or _flash_attn_forward is None:
        # Fallback simple : on applique juste l'attention hybride sur tout
        # On concatène img + texte et on laisse hybrid_seq_parallel_attn gérer.
        attn = hybrid_seq_parallel_attn(
            None,
            q,
            k,
            v,
            dropout_p=0.0,
            causal=False,
            joint_tensor_query=None,
            joint_tensor_key=None,
            joint_tensor_value=None,
            joint_strategy="rear",
        )
        b, s, h, d = attn.shape
        return attn.reshape(b, s, -1)

    # Cas "normal" prévu par HunyuanVideo (GPU massif / cluster)
    attn1 = hybrid_seq_parallel_attn(
        None,
        q[:, :img_q_len, :, :],
        k[:, :img_kv_len, :, :],
        v[:, :img_kv_len, :, :],
        dropout_p=0.0,
        causal=False,
        joint_tensor_query=q[:, img_q_len : cu_seqlens_q[1]],
        joint_tensor_key=k[:, img_kv_len : cu_seqlens_kv[1]],
        joint_tensor_value=v[:, img_kv_len : cu_seqlens_kv[1]],
        joint_strategy="rear",
    )

    softmax_scale = q.shape[-1] ** (-0.5)

    if flash_attn.__version__ >= "2.7.0":
        attn2, *_ = _flash_attn_forward(
            q[:, cu_seqlens_q[1] :],
            k[:, cu_seqlens_kv[1] :],
            v[:, cu_seqlens_kv[1] :],
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )
    else:
        attn2, *_ = _flash_attn_forward(
            q[:, cu_seqlens_q[1] :],
            k[:, cu_seqlens_kv[1] :],
            v[:, cu_seqlens_kv[1] :],
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=False,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )

    attn = torch.cat([attn1, attn2], dim=1)
    b, s, h, d = attn.shape
    attn = attn.reshape(b, s, -1)

    return attn
