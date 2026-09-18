# Normalization Techniques

## Motivation

Good [weight initialization](/topic/weight-initialization) sets a network's activations to a sensible scale on day one. But as training progresses and weights keep updating, that scale can drift — a layer's outputs might grow, shrink, or shift as everything feeding into it keeps changing. Each layer ends up training against a constantly moving target, which slows learning down and reopens the door to the shrinking or exploding gradients described in [Gradient Instability](/topic/gradient-instability) — this time as an ongoing problem during training, not just a bad starting point. Normalization techniques fix this by actively rescaling activations back to a consistent range at every step, not just at initialization.

## The core idea

At its simplest, normalizing a set of values means subtracting their mean ($\mu$) and dividing by their standard deviation ($\sigma$), so they end up centered around 0 with a consistent spread:

$$
\hat{x} = \frac{x - \mu}{\sigma}
$$

Doing this to a layer's activations keeps every layer feeding the next one a distribution that doesn't wander as training goes on. Most methods also add a small learnable scale $\gamma$ and shift $\beta$ afterward ($\gamma \hat{x} + \beta$), so the network can undo the normalization if that turns out to work better for a specific layer — normalization sets a stable default, the learnable parameters give the network room to deviate from it on purpose.

What differs between methods is *which* values get grouped together to compute $\mu$ and $\sigma$.

## Batch Normalization

**Batch Norm** computes $\mu$ and $\sigma$ across all examples in the current mini-batch, separately for each feature. It was the original version of this idea, and it works well for standard feedforward and convolutional networks — but it has a real weakness: its statistics depend on the batch itself, so it behaves inconsistently with very small batches, and needs special handling at inference time when there's no "batch" to normalize over (typically a running average collected during training).

## Layer Normalization

**Layer Norm** computes $\mu$ and $\sigma$ across all the features *within a single example* instead, completely independent of batch size. That makes it a natural fit for sequence models and Transformers, where batch size varies and sequences are processed one token at a time — it's the default normalization in nearly every modern large language model architecture.

## RMSNorm

**RMSNorm** simplifies Layer Norm further: instead of centering *and* rescaling, it only rescales, using the root-mean-square of the values instead of subtracting the mean first:

$$
\hat{x} = \frac{x}{\text{RMS}(x)}, \qquad \text{RMS}(x) = \sqrt{\frac{1}{n}\sum x_i^2}
$$

Dropping the mean-centering step turns out to barely affect performance in practice, while being noticeably cheaper to compute — which matters at the scale modern LLMs run at. This is why RMSNorm has replaced Layer Norm in several recent large models, including Llama.

## Further reading

Sergey Ioffe and Christian Szegedy introduced Batch Normalization in *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift* (2015). Jimmy Ba, Jamie Ryan Kiros, and Geoffrey Hinton introduced Layer Normalization in *Layer Normalization* (2016). Biao Zhang and Rico Sennrich introduced RMSNorm in *Root Mean Square Layer Normalization* (2019).