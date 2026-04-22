"""Register torch.ops.ggml.gated_delta_net — fused gated delta net step.

Maps to llama.cpp's ggml_gated_delta_net: a single CUDA kernel that computes
the recurrent delta-rule step (state decay + Sk + delta update + output) for
Mamba-style linear attention. Replaces ~80 ops per GDN layer.

Inputs (all F32, contiguous):
    q:     [B, T, H_k, K]  — query
    k:     [B, T, H_k, K]  — key
    v:     [B, T, H_v, V]  — value (H_v = H_k * head_repeat)
    g:     [B, T, H_v]     — scalar gate (will be unsqueezed internally)
    beta:  [B, T, H_v]     — delta rule weight (will be unsqueezed internally)
    state: [B, H_v, V, K]  — recurrent state (V == K for Qwen3.5)

Returns:
    output:    [B, T, H_v, V]  — new output
    new_state: [B, H_v, V, K]  — updated state (copy back into state buffer)
"""

import torch

ggml_lib = torch.library.Library("ggml", "FRAGMENT")
ggml_lib.define(
    "gated_delta_net(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor state) -> (Tensor, Tensor)"
)


@torch.library.impl(ggml_lib, "gated_delta_net", "CompositeExplicitAutograd")
def gated_delta_net_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
):
    """Eager fallback: recurrent delta-rule step."""
    B, T, H_k, K = q.shape
    _, _, H_v, V = v.shape
    rep = H_v // H_k
    scale = 1.0 / (V ** 0.5)

    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    g_f = g.float()
    beta_f = beta.float()
    state_f = state.float().clone()

    output = torch.empty(B, T, H_v, V, dtype=torch.float32, device=q.device)
    for t in range(T):
        q_t = q_f[:, t]  # [B, H_k, K]
        k_t = k_f[:, t]
        v_t = v_f[:, t]  # [B, H_v, V]
        g_t = g_f[:, t]  # [B, H_v]
        b_t = beta_f[:, t]

        if rep > 1:
            q_t = q_t.repeat_interleave(rep, dim=1)  # [B, H_v, K]
            k_t = k_t.repeat_interleave(rep, dim=1)

        decay = torch.exp(g_t).unsqueeze(-1).unsqueeze(-1)  # [B, H_v, 1, 1]
        state_f = state_f * decay
        Sk = torch.matmul(state_f, k_t.unsqueeze(-1)).squeeze(-1)  # [B, H_v, V]
        delta = b_t.unsqueeze(-1) * (v_t - Sk)  # [B, H_v, V]
        state_f = state_f + delta.unsqueeze(-1) * k_t.unsqueeze(-2)
        out_t = torch.matmul(state_f, q_t.unsqueeze(-1)).squeeze(-1) * scale
        output[:, t] = out_t.to(output.dtype)

    return output.to(q.dtype), state_f.to(state.dtype)


@torch.library.impl_abstract("ggml::gated_delta_net")
def gated_delta_net_abstract(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
):
    B, T, _, _ = q.shape
    _, _, H_v, V = v.shape
    _, _, _, K = state.shape
    output = q.new_empty((B, T, H_v, V), dtype=torch.float32)
    new_state = state.new_empty((B, H_v, V, K), dtype=torch.float32)
    return output, new_state
