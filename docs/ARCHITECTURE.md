# Architecture and objective

## Scope

This document describes only the frozen `sparsegen_shared_lamsparse_balance` arm. It does not describe the dense control, fixed Top-*k* ablations, router-CE arms, PPO, or earlier experimental variants.

The MoE is source agnostic. The router receives the decoder hidden state only; it is not given source, language, writer, or script metadata. There are four routed experts as a capacity budget, not as named language experts.

## MLP replacement

At decoder MLP layers \(\{1,5,9,13,17,21\}\), the native MLP is replaced by:

- four routed MLP copies \(E_{l,1},\ldots,E_{l,4}\);
- one shared MLP copy \(S_l\), always active;
- a linear router and an auxiliary hidden-to-scalar \(\lambda\) network.

All five MLPs are deep copies of the inherited dense MLP. The router is initialized close to uniform. Since the routed weights sum to one and every MLP copy initially represents the same function, the replacement begins as the inherited dense MLP up to numerical precision:

\[
(1-\alpha)\sum_e p_e E(h) + \alpha E(h)=E(h),\quad \alpha=0.25.
\]

This is an initialization-stability statement, not evidence that the experts remain identical after training.

## Native Sparsegen router

For each non-padding token-layer position, a linear router produces four logits. The implementation applies a Sparsegen transformation with a learned, token-specific \(\lambda\). It can assign an exact zero to one or more routed experts. There is no hard Top-*k* constraint in this arm.

\[
p_{l,t}=\operatorname{Sparsegen}(r_l(h_{l,t}),\lambda_l(h_{l,t})),\qquad
\sum_{e=1}^{4}p_{l,t,e}=1,\quad p_{l,t,e}\ge0.
\]

The routed and shared branches are fused as:

\[
\tilde h_{l,t}=0.75\sum_{e=1}^{4}p_{l,t,e}E_{l,e}(h_{l,t})+0.25S_l(h_{l,t}).
\]

The Python workflow dispatches each routed expert only to token positions with positive gate weight. The shared branch runs at every token position. No capacity factor, token dropping, or queueing policy is implemented. Therefore, report measured latency rather than extrapolating end-to-end compute savings from gate sparsity.

## Objective

The training wrapper computes answer-only causal cross entropy in FP32. It upweights the terminal token of `page_htr` targets by 4.0 and normalizes gradients by the total weighted answer-token normalizer across the complete accumulation window. This makes the token objective invariant to the partition into uneven microbatches.

\[
\mathcal L = \mathcal L_{\mathrm{LM}}+
\gamma_{\lambda}\mathcal L_{\mathrm{LamSparse}}+
\gamma_{\mathrm{bal}}\mathcal L_{\mathrm{balance}},
\quad \gamma_{\lambda}=\gamma_{\mathrm{bal}}=0.01.
\]

### LamSparse

The lambda branch ends in `LogSigmoid`; after the implementation offset, \(\lambda<1\). The raw regularizer is exactly:

\[
\mathcal L_{\mathrm{LamSparse}}=\operatorname{mean}(1-\lambda).
\]

Minimizing it raises \(\lambda\), which increases the sparsity pressure in the native Sparsegen calculation. It regularizes only the routed support; the always-active shared expert is not part of this support.

### Routed-expert load balance

For routed-expert mass \(p\) and a realized positive-gate indicator \(a=\mathbb{1}[p>0]\), the implementation calculates, per selected layer:

\[
\operatorname{prop}_e=\frac{1}{N}\sum_t p_{t,e},\qquad
\operatorname{freq}_e=\frac{\sum_t a_{t,e}}{\sum_{t,j}a_{t,j}},
\]

\[
\mathcal L_{\mathrm{balance}}=4\sum_{e=1}^{4}\operatorname{prop}_e\operatorname{freq}_e.
\]

The loss is averaged across selected MoE layers and excludes the shared expert. It is Switch-style importance-times-realized-activation regularization, but uses the actual positive-gate frequency appropriate to variable Sparsegen support. It is not a router classification loss.

Both MoE regularizers linearly warm from zero to their stated coefficient over the first 10% of optimizer updates. The warmup prevents a copied-expert module from receiving full sparsity/balance pressure at its first update.

## Explicitly disabled mechanisms

| Mechanism | Released value | Interpretation |
| --- | ---: | --- |
| Router CE | 0 | No metadata supervision for routes. |
| Token unlikelihood | 0 | Not part of the comparison. |
| Fixed Top-*k* | Off | Sparsegen controls support size dynamically. |
| Source/group replacement sampler | Off | One full pass over the train records per epoch. |
| Augmentation | Off | No unrecorded image perturbation. |
| Bbox prediction | Off | Structural supervision is page line count only. |

## What the method does not establish

- It does not identify pure script-specific experts. The observed routes may reflect source, handwriting style, layout, text length, and scripts jointly.
- It does not prove that the manually spaced six layers are optimal. A same-budget placement ablation is required for that claim.
- It does not imply fixed Top-2/Top-4 behavior. Support size must be measured from the checkpoint.
- It does not establish speedup without a controlled latency or FLOP study.
