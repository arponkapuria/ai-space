# Self-Supervised Learning

## Motivation

[Autoencoders](/topic/autoencoders) already sidestep one real problem: labeled data is expensive, but raw unlabeled data is everywhere. An autoencoder uses the input as its own target. Self-supervised learning generalizes that trick: hide or alter *some part* of the input, and train the model to recover it. The hidden part was already sitting in the original, unmodified data — a free training target, no human labeling required.

## Where the "labels" actually come from

Ordinary supervised learning needs (input, label) pairs where a person decided the label. Self-supervised learning instead builds a fill-in-the-blank task directly from ordinary, already-existing data — no person involved. Two examples:

- **Text:** "the cat sat on the \_\_\_" — hide "mat," predict it from context. The answer was already in the sentence; nobody annotated it. This is [BERT](/topic/bert)'s masked language modeling.
- **Images:** rotate an unlabeled photo by 0°, 90°, 180°, or 270°, and predict which rotation was applied. The label is known automatically, since the training code applied it.

In both cases, something already in the raw data becomes the target, for free — that's what "self-supervised" means.

## Two families of pretext task

The artificial task built this way is a **pretext task** — not useful on its own, but solving it forces the model to learn something genuinely useful. Two families:

**Contrastive tasks** train the model to recognize when two things come from the same source versus different ones, rather than reconstructing anything. This has its own lineage (SimCLR → MoCo → BYOL → DINO) and loss function, covered in [Contrastive Learning](/topic/contrastive-learning).

**Generative / reconstructive tasks** predict or rebuild missing content directly. Before the field settled on one dominant recipe, researchers tried several hand-designed versions:

- **Rotation Prediction.** Predict a random 0°/90°/180°/270° rotation. Cheap, but exploitable — a fixed visual cue (a watermark, always-upright text) lets the model cheat instead of reasoning about shape.
- **Jigsaw Puzzles.** Shuffle an image's patches, predict the correct arrangement — forces learning how parts relate spatially.
- **Colorization.** Predict a grayscale image's original colors — draws on real semantic knowledge (bananas are yellow), stronger signal but limited to what color carries.
- **Inpainting.** Reconstruct a masked-out chunk of the image — the direct ancestor of masked prediction.

**Loss function depends on what's predicted.** Rotation and Jigsaw are classification in disguise ("which of 4 rotations") — **cross-entropy**. Colorization and Inpainting predict continuous pixel values — **mean squared error**. Masked word prediction is discrete too, just over the whole vocabulary — cross-entropy again. MAE (below) predicts raw pixels — MSE again. Same rule as everywhere else: discrete choice → cross-entropy, continuous value → MSE.

Each hand-designed task only captures a narrow slice of "understanding," with its own shortcut risk. **Masked prediction** — hide a random chunk, predict it directly, no special-case design — turned out far more general, working almost unchanged for text (BERT) and images (MAE), and scaling better than the hand-crafted alternatives. That's why it became dominant. One detail worth knowing: MAE masks ~75% of an image's patches, while BERT masks only ~15% of a sentence's words — images are spatially redundant (neighboring pixels correlate heavily), text isn't, so it needs a much lower ratio to stay solvable.

## Frameworks, at a glance

| Framework | Domain | Core approach |
|---|---|---|
| MAE (Masked Autoencoder) | Vision | Mask image patches, reconstruct them directly |
| BERT (MLM) | Text | Mask tokens, predict them from context |
| GPT (autoregressive) | Text | Predict the next token from everything before it |
| SimCLR / MoCo / BYOL / DINO | Vision | Contrastive — see [Contrastive Learning](/topic/contrastive-learning) |

GPT's next-token prediction is self-supervised too — the "label" for each position is just the next word already in the training corpus, no contrastive machinery needed.

## How this is used in practice

Pretrain on a huge pile of unlabeled data with a pretext task, then fine-tune or apply [Transfer Learning](/topic/transfer-learning) on a much smaller labeled dataset for the actual task. This "pretrain, then adapt" pattern dominates modern NLP (BERT, GPT) and increasingly vision (DINOv2, MAE-pretrained backbones). Before a full fine-tune, a cheap sanity check is the linear probe from [Representation Learning](/topic/representation-learning) — freeze the model, train only a small classifier on top, and see how the representation holds up alone.

## Further reading

Mehdi Noroozi and Paolo Favaro introduced the jigsaw pretext task in *Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles* (2016). Richard Zhang, Phillip Isola, and Alexei Efros introduced colorization as a pretext task in *Colorful Image Colorization* (2016). Deepak Pathak et al. introduced inpainting-based pretraining in *Context Encoders: Feature Learning by Inpainting* (2016). Spyros Gidaris, Praveer Singh, and Nikos Komodakis introduced rotation prediction in *Unsupervised Representation Learning by Predicting Image Rotations* (2018). Kaiming He et al. introduced Masked Autoencoders (MAE) in *Masked Autoencoders Are Scalable Vision Learners* (2021).