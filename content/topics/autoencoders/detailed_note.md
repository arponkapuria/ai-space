# Autoencoders

## Motivation

[Encoder-Decoder](/topic/encoder-decoder) architectures compress an input into a representation, then decode that representation into some output. An autoencoder is the simplest possible version of this: set the target output $y$ to be the *same thing* as the input $x$. The network's only job is to reconstruct exactly what it was given. That might sound pointless at first — why train a network to output what it already has? — but forcing that reconstruction through a narrow bottleneck is precisely what makes it useful: if the network can still rebuild the input accurately after being squeezed down, whatever the bottleneck kept must have captured what actually matters. And because the target is just the input itself, no labels are needed at all — any raw data works.

## Why the bottleneck matters

If the encoder's output $z$ were allowed to be the same size as the input (or larger), the easiest solution is to just learn the identity function — copy the input straight through, encoder and decoder doing nothing useful. The bottleneck is what prevents that: $z$ has to be small enough that copying isn't possible, forcing the network to prioritize and compress instead. Training just minimizes the difference between the input $x$ and the reconstruction $\hat{x}$ — commonly mean squared error.

## Variants, and when each is used

**When to reach for a plain autoencoder:** rarely, on its own — it's the baseline the variants below improve on for almost any real use case.

- **Denoising Autoencoder.** Trained on a *corrupted* version of the input, with the *clean* version as the target. Since copying the corrupted input straight through would reproduce the noise, not the clean signal, the network is forced to learn what the underlying structure actually looks like, rather than just compressing without necessarily understanding anything. Useful directly for image denoising, and historically used as a pretraining step before modern self-supervised methods took over that role.
- **Sparse Autoencoder.** Instead of shrinking the bottleneck's *size*, this keeps the layer wide but adds a penalty that pushes most neurons toward zero for any given input — only a handful stay active at once. This achieves a similar forced-compression effect through a different mechanism, and it's the technique behind modern **sparse autoencoders used for LLM interpretability**, where a wide, sparse latent space makes it possible to isolate individual, human-interpretable concepts a model has learned.
- **Variational Autoencoder (VAE).** Instead of encoding an input to one fixed point, a VAE encodes it to a small probability distribution (a mean and variance), and decodes from a sample drawn from that distribution. Because nearby points in latent space now decode to similar, sensible outputs, sampling a brand-new point and decoding it produces a plausible, novel output — turning an autoencoder from a compression tool into a generative model. Covered in more depth in [VAE](/topic/vae).

## Where this actually shows up in practice

- **Anomaly detection.** Train only on "normal" examples; at inference time, anything the model reconstructs poorly (high reconstruction error) is flagged as unusual. Used for fraud detection and manufacturing defect detection, among others — genuinely common in production, and one of the few places plain autoencoders are still a first choice rather than a stepping stone to something else.
- **Nonlinear dimensionality reduction.** An alternative to PCA when the underlying structure in the data isn't well captured by a linear projection.
- **Denoising**, directly, as described above.
- **Generation**, via VAEs specifically — though for image generation, VAEs have mostly been overtaken by [Diffusion Models](/topic/diffusion-models), which produce sharper results; VAEs still show up as a component *inside* some diffusion pipelines (compressing images to a smaller latent space before the diffusion process runs there instead of on raw pixels).

## Further reading

Geoffrey Hinton and Ruslan Salakhutdinov's *Reducing the Dimensionality of Data with Neural Networks* (2006) is the paper most credited with popularizing deep autoencoders for dimensionality reduction. Pascal Vincent et al. introduced the Denoising Autoencoder in *Extracting and Composing Robust Features with Denoising Autoencoders* (2008). Diederik Kingma and Max Welling introduced the Variational Autoencoder in *Auto-Encoding Variational Bayes* (2013).