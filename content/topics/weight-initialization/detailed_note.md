# Weight Initialization

## Motivation

[Gradient Instability](/topic/gradient-instability) happens when the chain of multiplications in [Backpropagation](/topic/backpropagation) shrinks or explodes across many layers. That chain starts moving in the wrong direction from the very first step if the weights it's built on start out at the wrong scale — so before any training happens at all, the values you pick to start each weight at already determine whether the network has a fighting chance.

## Why the starting values matter

A few naive choices fail in predictable ways:

| Scheme | What it does | What goes wrong |
|---|---|---|
| **Zero init** | Every weight starts at exactly 0 | Every neuron in a layer computes the same output and gets the same gradient, so they all update identically forever — the layer behaves like it has one neuron no matter how many you gave it |
| **Naive random init** | Weights drawn randomly (e.g. uniform or standard normal), without accounting for layer size | Breaks the zero-init symmetry problem, but with no scale correction, activations tend to grow or shrink layer over layer — exploding or vanishing, depending on the network's depth and width |
| **Xavier / He init** | Weights drawn randomly, but scaled based on the layer's size (below) | This is the fix — random enough to break symmetry, scaled correctly enough to avoid exploding or vanishing |

The fix is a **scaled** random start: random enough to break symmetry between neurons, but scaled to the size of the layer so the signal doesn't grow or shrink as it passes through.

## Getting the scale right

The right scale depends on **fan-in** (how many inputs feed into a neuron) and **fan-out** (how many neurons it feeds into) — because a layer with more inputs naturally sums up more terms, so each individual weight needs to be smaller to keep the total sum in a reasonable range.

**Xavier (Glorot) initialization**, designed for sigmoid and tanh, sets each weight's variance to:

$$
\text{Var}(w) = \frac{2}{\text{fan\_in} + \text{fan\_out}}
$$

**He initialization**, designed for ReLU, uses a larger variance:

$$
\text{Var}(w) = \frac{2}{\text{fan\_in}}
$$

The difference exists because ReLU zeroes out roughly half its inputs (everything negative), so the variance needs to be doubled to compensate for that lost signal and keep activations at a consistent scale layer to layer. Using Xavier init with ReLU networks (or vice versa) tends to make gradients shrink or grow slightly faster than intended — not catastrophic, but a measurable drag on how quickly training gets going.

## In practice

Every major framework picks a sensible default automatically — PyTorch's linear and convolutional layers use a variant of He initialization by default, for instance — so this usually isn't something you configure by hand. It matters most when building a custom architecture from scratch, or diagnosing why a network trains unusually slowly or produces `NaN` losses right at the start, since a bad initialization is one of the first things worth ruling out.

Good initialization only guarantees good *starting* behavior. It doesn't keep gradients well-scaled as training progresses and weights drift away from their initial values — that ongoing job belongs to [Normalization Techniques](/topic/normalization-techniques).

## Further reading

Xavier Glorot and Yoshua Bengio introduced Xavier initialization in *Understanding the Difficulty of Training Deep Feedforward Neural Networks* (2010). Kaiming He et al. introduced He initialization, tailored for ReLU, in *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification* (2015).