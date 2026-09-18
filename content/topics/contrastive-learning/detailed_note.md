# Contrastive Learning

## Motivation

[Self-Supervised Learning](/topic/self-supervised-learning) splits its pretext tasks into two families: reconstructing hidden content directly, or learning by *comparison* instead. Contrastive learning is the second path — rather than training a model to rebuild a masked word or a blocked-out patch, it trains the model to recognize when two things come from the same underlying source versus different ones. This turns out to be a remarkably effective way to learn representations, without ever needing the model to generate or reconstruct anything at all.

## How it works, concretely

Take one unlabeled image. Create two different *views* of it using random augmentation — a random crop, a color shift, a flip. Since both views came from the same original image, this is a **positive pair**. Now take a different image from the same batch and augment it too — relative to the first image, this is a **negative**. The training goal: push the two views of the *same* image close together in representation space, while pushing views of *different* images apart. This happens for every image in the batch simultaneously, all at once.

**Why this actually teaches something useful:** to correctly recognize "these two crops came from the same photo" across random cropping, color changes, and flips, the model can't rely on exact pixel values — those changed between the two views. It's forced to learn whatever *doesn't* change: the actual content, the object, the shape — precisely the kind of transferable representation [Representation Learning](/topic/representation-learning) is after. Augmentation choice matters enormously here: augmentations that are too weak leave shortcuts the model can exploit instead of learning real content (e.g. matching by leftover color cast), while augmentations that are too aggressive can destroy the very thing that made the pair "positive" in the first place.

## The loss function

In plain terms first: for one image's view, the loss is a softmax over the entire batch — "out of every other view in this batch, correctly pick out the one true positive, and make it score higher than all the negatives."

<details>
<summary>The contrastive loss (InfoNCE / NT-Xent)</summary>

Formally, for a positive pair $(i, j)$ among $N$ examples in a batch:

$$
\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k)/\tau)}
$$

$\text{sim}(z_i, z_j)$ measures how similar two representations are (commonly cosine similarity), and $\tau$ is a temperature that controls how sharply the model has to separate the true positive from everything else — a lower temperature pushes the model to make sharper, more confident distinctions. The denominator sums similarity against every *other* item in the batch, and every one of those acts as a negative.

</details>

## The negatives problem, and how the field moved past it

This loss needs a lot of negatives per positive to work well — the more things a positive has to be distinguished from, the more the model actually has to learn. This single practical constraint shaped almost the entire lineage of methods that followed:

- **SimCLR** (2020) is the most direct implementation of the approach above, and it needs it literally: very large batch sizes, thousands of images at once, so there are enough negatives in every batch to make the comparison meaningful. Effective, but expensive.
- **MoCo** (2020) decouples negative count from batch size by keeping a running *queue* of representations from recent batches to compare against, updated by a slowly-moving [EMA](/topic/weight-averaging) encoder rather than recomputed from scratch every step. Same core idea, far cheaper to run.
- **BYOL** (2020) removes negatives entirely — a genuine surprise at the time, since negatives were assumed necessary to prevent the trivial shortcut of the model just collapsing everything to the same output. BYOL uses two networks instead: an online network being trained, and a target network that's an EMA copy of the online one. The online network is trained only to predict the target network's output for a differently-augmented view of the same image — no negatives, no contrasting against other images at all, and it still works.
- **DINO** (2021) builds on the same EMA-teacher idea as BYOL, framed as self-distillation, and turned out to work especially well for Vision Transformers — it's part of why DINO-pretrained ViTs became a popular general-purpose vision backbone.

The overall trend across this lineage is a steady move away from needing large numbers of explicit negatives at all, trading it for architectural tricks (a slow-moving teacher network) that prevent the same collapse problem a different way.

## Extending the idea across modalities: CLIP

Every method above builds its positive pairs from two augmented views of the *same image*. **CLIP** applies the same core idea across a completely different kind of pair: an image and its actual caption, scraped from the internet, become the positive pair — while every other (image, caption) combination in the batch is a negative, exactly like before. The model learns to pull a photo and its real caption close together in a shared representation space, while pushing every mismatched pairing apart. This is still the same InfoNCE-style contrastive loss described above, just applied across two different modalities instead of two augmented views of one. It's covered in full in [CLIP](/topic/clip).

## Picking one in practice

| Method | Needs negatives? | Main cost |
|---|---|---|
| SimCLR | Yes, many | Very large batch sizes |
| MoCo | Yes, but decoupled from batch size | Maintaining a representation queue |
| BYOL | No | Two networks, one EMA-updated |
| DINO | No | Two networks, one EMA-updated; tuned for Vision Transformers |
| CLIP | Yes, many | Very large batch sizes, like SimCLR — but across image-text pairs |

MoCo and its descendants are generally preferred over plain SimCLR today specifically because they get comparable results without needing the largest, most expensive batch sizes.

## Further reading

Aaron van den Oord, Yazhe Li, and Oriol Vinyals introduced the InfoNCE loss in *Representation Learning with Contrastive Predictive Coding* (2018). Ting Chen et al. introduced SimCLR in *A Simple Framework for Contrastive Learning of Visual Representations* (2020). Kaiming He et al. introduced MoCo in *Momentum Contrast for Unsupervised Visual Representation Learning* (2020). Jean-Bastien Grill et al. introduced BYOL in *Bootstrap Your Own Latent* (2020). Mathilde Caron et al. introduced DINO in *Emerging Properties in Self-Supervised Vision Transformers* (2021). Alec Radford et al. introduced CLIP in *Learning Transferable Visual Models From Natural Language Supervision* (2021).