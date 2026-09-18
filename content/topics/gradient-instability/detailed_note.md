# Gradient Instability

## Motivation

[Backpropagation](/topic/backpropagation) computes a weight's gradient by multiplying together a chain of local effects, one per layer, stretching from the loss back to that weight. That multiplication is harmless for a shallow network with just a couple of layers. But push the network deeper — the whole point of deep learning — and multiplying many numbers together in a row starts to misbehave in one of two ways: the product can shrink toward zero, or it can grow explosively large. Both make training fail, just in opposite directions, and both showed up as soon as people tried to train genuinely deep networks.

## Vanishing gradients

If each local derivative in the chain is a bit smaller than 1 — which is exactly what happens with [sigmoid or tanh](/topic/activation-functions), whose derivatives max out well below 1 — then multiplying many of them together shrinks fast. A chain of 10 numbers around 0.2 each multiplies down to about 0.0000001. In a deep network, that means the gradient reaching the earliest layers is practically zero: those layers stop learning almost entirely, even while later layers keep training normally. The network doesn't crash — it just quietly stalls, with its earliest layers stuck near their random starting values.

## Exploding gradients

The opposite happens if the local derivatives are consistently larger than 1. The product grows instead of shrinking, and it grows fast — the same compounding effect, just in reverse. Weight updates become enormous, the loss swings wildly instead of settling down, and values can overflow into `NaN`, breaking training outright.

<details>
<summary>Why depth makes this so much worse</summary>

If every local derivative in a chain of $n$ layers is roughly some value $r$, the combined gradient scales like $r^n$. For $r$ slightly below 1, $r^n$ shrinks exponentially as $n$ grows. For $r$ slightly above 1, $r^n$ grows exponentially instead. This is exactly like compound interest — a tiny difference in the per-step rate barely matters for a few steps, but compounded over dozens or hundreds of layers, it's the difference between "basically zero" and "basically infinite." A shallow network never really hits this regime; a deep one hits it by default unless something specifically counteracts it.

</details>

## Fixes

No single fix solves this — modern deep networks combine several:

- **[Activation functions](/topic/activation-functions) with a well-behaved derivative.** ReLU's derivative is exactly 1 for positive inputs, so it doesn't shrink the chain the way sigmoid does. This is a big part of why ReLU replaced sigmoid in hidden layers.
- **[Weight initialization](/topic/weight-initialization).** Starting weights at the right scale (not too big, not too small) keeps the chain's per-layer multiplier close to 1 from the very first step, instead of drifting into shrink-or-explode territory immediately.
- **[Normalization techniques](/topic/normalization-techniques)** like batch norm or layer norm rescale activations between layers, which keeps gradients flowing at a consistent scale throughout training, not just at initialization.
- **Skip connections**, which let the gradient bypass a layer entirely by adding an unmodified copy of the input back in — so even if a particular layer's local derivative is small, the gradient still has a direct path through, without that multiplication shrinking it.
- **Gradient clipping**, used specifically for exploding gradients — if the gradient's size exceeds some threshold during training, it just gets scaled back down before the weight update is applied.

## Further reading

Sepp Hochreiter's 1991 diploma thesis is generally credited as the first clear identification of the vanishing gradient problem, with Yoshua Bengio, Patrice Simard, and Paolo Frasconi's *Learning Long-Term Dependencies with Gradient Descent is Difficult* (1994) formalizing it further. Xavier Glorot and Yoshua Bengio's *Understanding the Difficulty of Training Deep Feedforward Neural Networks* (2010) and Kaiming He et al.'s *Delving Deep into Rectifiers* (2015) both address it through principled weight initialization.