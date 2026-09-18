# Activation Functions

## Motivation

A [Perceptron](/topic/perceptron)'s hard step function is what makes it a classifier, but it has a fatal flaw for anything deeper: its derivative is zero almost everywhere, so there's no gradient to push backward through multiple layers. The [MLP](/topic/mlp) fixes this by swapping the step function for something smooth and differentiable — but "smooth" still left a lot of room for different choices, and the history of activation functions is really a history of each choice's own new problem showing up once networks got deeper. Sigmoid and tanh came first, worked fine for shallow networks, then quietly broke as depth increased. ReLU fixed that, and introduced its own failure mode. GELU and Swish exist to fix ReLU's. Every step in this chain is a direct response to a training problem the previous function caused.

## Sigmoid and tanh

The earliest choices squash any input into a bounded range:

$$
\sigma(x) = \frac{1}{1 + e^{-x}} \in (0, 1), \qquad \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \in (-1, 1)
$$

Both are smooth and easy to interpret — sigmoid's output looks like a probability, which is why it's still used at an output layer for binary classification. Tanh is just a rescaled sigmoid, centered at zero instead of 0.5, which tends to make training a bit better behaved since its outputs aren't all positive.

The problem is **saturation**. For large positive or negative inputs, both functions flatten out — their slope approaches zero. During backpropagation, that near-zero slope gets multiplied in at every layer the gradient passes through. Stack enough layers and the gradient reaching the earliest layers shrinks toward nothing, so those layers barely update. This is the **[vanishing gradient problem](/topic/gradient-instability)**, and it's the main reason sigmoid and tanh are rarely used in hidden layers of deep networks today, even though they're still common at output layers where the bounded range is the whole point.

## ReLU and Leaky ReLU

**ReLU** (Rectified Linear Unit) is almost aggressively simple:

$$
\text{ReLU}(x) = \max(0, x)
$$

For any positive input, the gradient is exactly 1 — no shrinking, no saturation on that side. That single property is most of why ReLU became the default activation for hidden layers: gradients pass through cleanly for active units, training is faster, and the function itself costs almost nothing to compute.

The trade-off is the flip side of the same design: for any negative input, both the output and the gradient are exactly zero. If a unit's weights drift so its input is always negative, it stops contributing anything and never updates again — it's permanently "dead." This is the **dying ReLU problem**, and it's common enough in practice that a small fix exists: **Leaky ReLU**, which allows a small nonzero slope for negative inputs instead of flattening to zero entirely:

$$
\text{LeakyReLU}(x) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases}
$$

where $\alpha$ is a small constant (commonly 0.01). It keeps a trickle of gradient alive for negative inputs, so a unit that drifts negative has a chance to recover instead of dying permanently.

## GELU and Swish

ReLU has a sharp corner at zero — its derivative jumps abruptly from 0 to 1. That's fine for many networks, but it turns out to matter for the very large, very deep architectures used in modern NLP and vision. **GELU** (Gaussian Error Linear Unit) and **Swish** are smooth alternatives that behave similarly to ReLU for large inputs but curve gently through zero instead of bending sharply:

$$
\text{GELU}(x) = x \cdot \Phi(x), \qquad \text{Swish}(x) = x \cdot \sigma(x)
$$

where $\Phi(x)$ is the standard normal cumulative distribution function. Both multiply the input by something close to a smoothed step function, which lets small negative inputs pass through slightly instead of being zeroed out entirely — a softer version of what Leaky ReLU does, but differentiable everywhere. GELU is the default activation inside most modern Transformer-based models (BERT, GPT-style architectures), largely because that extra smoothness measurably helps optimization at the scale these models train at, even though it costs a bit more compute per activation than plain ReLU.

## Picking the right one

| | Use when |
|---|---|
| **Sigmoid** | Output layer for binary classification, where you need a 0–1 probability |
| **Softmax** (sigmoid's multi-class extension) | Output layer for multi-class classification |
| **Tanh** | Occasionally in RNN gates; rare elsewhere today |
| **ReLU** | Default for hidden layers in CNNs and simpler feedforward networks — fast, cheap, works well |
| **Leaky ReLU** | Same situations as ReLU, if dying units show up in practice |
| **GELU / Swish** | Hidden layers in large Transformer-based architectures, where the extra smoothness pays off |

## Further reading

Sigmoid and tanh have roots in classical neuroscience-inspired modeling going back decades. ReLU's modern popularity traces largely to Nair and Hinton's *Rectified Linear Units Improve Restricted Boltzmann Machines* (2010) and its use in AlexNet (2012). GELU comes from Hendrycks and Gimpel's *Gaussian Error Linear Units* (2016), and Swish from Ramachandran, Zoph, and Le's *Searching for Activation Functions* (2017).