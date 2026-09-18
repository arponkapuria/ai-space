# Representation Learning

## Motivation

Classical ML largely depends on [Feature Engineering](/topic/feature-engineering) — a person deciding, by hand, which properties of the raw data are worth feeding into a model. That works, but it's slow, needs domain expertise, and has to be redone for every new problem. Deep networks fix part of this: an [MLP](/topic/mlp) or CNN learns its own internal features automatically, layer by layer, instead of being handed them. But trained the ordinary way, those features are learned from scratch for one specific task, on one specific labeled dataset, and thrown away once that task is done — even though many tasks actually share a lot of underlying structure. Edges and textures useful for classifying cats are also useful for detecting cars; the meaning of a word useful for translation is also useful for summarization. Representation learning reframes the goal: instead of training a model to solve one task, train it to produce a good general-purpose *encoding* of the data itself — one reusable across many tasks, not thrown away after one.

## What makes a representation "good"

A representation is just a vector — a list of numbers standing in for a raw input like an image or a sentence. The space these vectors live in is called the **latent space** ("latent" meaning hidden, not directly observed in the raw data), and "embedding" is the term used almost interchangeably with "representation" in practice — a word embedding, a sentence embedding, an image embedding are all just representations by another name.

Compressing something as complex as an image down to a short vector only works because of the **manifold hypothesis**: real data doesn't actually fill up the full high-dimensional space it's expressed in. The set of all possible pixel combinations is astronomically large, but the tiny sliver of those that look like an actual photograph sits on a much lower-dimensional surface within that space. A good representation isn't throwing information away arbitrarily — it's recovering that lower-dimensional structure that was already there.

This isn't a new idea unique to deep learning, either — [Dimensionality Reduction](/topic/dimensionality-reduction) techniques like PCA do exactly this with a linear projection instead of a neural network. Representation learning is the same goal, extended to nonlinear, learned projections instead of a fixed linear one.

A few properties decide whether a given latent space is actually useful:

- **Compact.** A good representation is far smaller than the raw input, capturing what matters and discarding what doesn't. This makes anything trained on top of it faster and less data-hungry, since it isn't relearning "what matters" from scratch.
- **Disentangled.** Different dimensions of the vector correspond to different, independent factors — one direction might track an object's rotation, another its lighting, without the two tangled together. This makes downstream tasks easier, since a model built on top doesn't have to untangle unrelated factors of variation first.
- **Transferable.** This is the property that matters most in practice, and it's directly testable: freeze the representation entirely, train only a small linear classifier on top of it for some new task, and see how well that does. A representation that lets a simple linear model perform well on a task it was never explicitly trained for — this technique is called a **linear probe** — is a genuinely good one. If only a full, expensive retraining of the whole thing can make it useful, the representation itself isn't doing much work.

## Ways to actually learn one

There's no single recipe — several different approaches all aim at this same goal, each covered in its own note:

- **[Autoencoders](/topic/autoencoders)** force a network to compress its input through a narrow bottleneck and reconstruct it, using whatever the bottleneck settles on as the representation.
- **[Self-Supervised Learning](/topic/self-supervised-learning)** creates its own training labels directly from the raw data — no human annotation needed — and learns a representation as a side effect of solving that self-generated task.
- **[Transfer Learning](/topic/transfer-learning)** skips learning a representation from scratch entirely, and instead reuses one already learned on a large, different dataset.
- Graph-structured data — social networks, molecules, knowledge graphs — needs representations of its own, which is what **[GNN](/topic/gnn)**-style architectures are built for.
- A more recent direction, **JEPA (Joint Embedding Predictive Architecture)**, proposed by Yann LeCun in 2022, takes a third path: instead of reconstructing raw pixels (like an Autoencoder) or contrasting augmented pairs of an input (like contrastive self-supervised learning), it predicts a missing or future part's *representation* directly in latent space, skipping the raw reconstruction step entirely. The idea is that predicting in latent space forces the model to focus on what actually matters and ignore irrelevant pixel-level detail. LeCun frames this specifically as a stepping stone toward **world models** — representations that capture enough about how an environment evolves to predict its future states, relevant to robotics and reinforcement learning, and covered further under [RL Foundations](/topic/rl-foundations).

## Further reading

Yoshua Bengio, Aaron Courville, and Pascal Vincent's *Representation Learning: A Review and New Perspectives* (2013) is the paper most credited with framing this as its own distinct goal, separate from any single task. Yann LeCun's *A Path Towards Autonomous Machine Intelligence* (2022) introduces JEPA and the broader world-model framing it's built toward.