# Generalization Techniques

## Motivation

A deep network with millions or billions of parameters has more than enough capacity to simply memorize its entire training set — hitting near-zero training loss while getting worse at anything it hasn't seen before. [Regularization](/topic/regularization) in classical ML (L1/L2 penalties) helps, but deep networks need more: specific techniques built around how neural networks actually train, layer by layer, epoch by epoch. If training loss keeps falling while validation loss climbs or flattens, that gap is the signal these techniques exist to close.

## Dropout

**When to reach for it:** the model's training accuracy is much higher than its validation accuracy, and the network is large relative to the amount of data.

**How it works:** during training, each forward pass randomly zeroes out a fraction of neurons (commonly 20–50%) in a layer, different neurons each time. This stops any single neuron from becoming a crutch the rest of the network over-relies on, since it might not be there next pass — the network is forced to learn redundant, more robust patterns instead. At test time, dropout is turned off and every neuron is active; outputs are rescaled to account for the fact that training only ever saw a fraction of the network active at once.

Dropout is common in classic CNNs and in the feedforward blocks of many Transformers, but it's frequently set to *zero* when pretraining large language models — at that scale, the dataset itself is so large relative to the model that memorization stops being the main risk, and the extra noise dropout introduces can just slow convergence down for no benefit. It's much more relevant when training on a comparatively small, fixed dataset.

## Weight Decay

**When to reach for it:** almost always on by default — it's cheap, and it's the standard companion to nearly every optimizer.

**How it works:** the same idea as L2 regularization — a small pull on every weight toward zero each step, discouraging any single weight from growing unnecessarily large, which tends to produce smoother, less overfit decision boundaries. As covered in [Optimizers](/topic/optimizers), the exact mechanism matters: **AdamW**'s whole reason for existing is applying this pull correctly, separately from Adam's adaptive per-weight scaling, instead of letting the two interact in a way that weakens the intended regularization.

## Early Stopping

**When to reach for it:** essentially every training run, as a cheap safety net — it costs nothing beyond tracking validation loss.

**How it works:** stop training once validation performance stops improving for a set number of epochs, instead of running a fixed number of epochs regardless. Training loss will often keep falling well past the point where validation loss bottoms out — that gap is the model starting to memorize training-specific noise instead of learning anything that generalizes further. Early stopping catches that point automatically.

## Data Augmentation

**When to reach for it:** the dataset is small or narrow relative to how much variation the model will see in the real world.

**How it works:** artificially expand the effective training set by applying realistic transformations that shouldn't change the label — random crops, flips, and color jitter for images; synonym replacement or back-translation for text; pitch or speed shifts for audio. A step further than transforming single examples, **Mixup** and **CutMix** blend *two* training examples together — Mixup linearly blends two images' pixels (and their labels, proportionally), while CutMix pastes a patch from one image onto another (with the label mixed by the patch's area). Both push the model to behave more linearly between classes instead of memorizing sharp, brittle boundaries, and they're common in modern image classification training recipes.

Its role shrinks at LLM pretraining scale for the same reason Dropout's does — with a large enough and diverse enough training corpus, the marginal benefit of synthetically varying the data further drops off, and the emphasis shifts to selecting and filtering existing data rather than augmenting it.

## Label Smoothing

**When to reach for it:** classification tasks where the model is being noticeably overconfident — very high softmax probabilities for the predicted class, even when it's wrong.

**How it works:** instead of training against a hard target (100% probability on the correct class, 0% everywhere else), the target is softened slightly (e.g. 90% on the correct class, the remaining 10% spread across the rest). This keeps the model from pushing its logits to extreme values to chase a target it can never actually reach exactly, which tends to improve calibration and generalization slightly. It's used in image classification training and shows up in some Transformer training recipes as well.

## A related but distinct idea: Weight Averaging

Techniques like SWA and EMA also improve a model's final quality by blending weights from several points in training instead of keeping only the last one — but they aren't really about preventing overfitting the way the techniques above are, so they get their own note: [Weight Averaging](/topic/weight-averaging).

## Combining them in practice

| Technique | Typical setting | Cost |
|---|---|---|
| Weight Decay | Nearly always on | Essentially free |
| Early Stopping | Nearly always on | Essentially free |
| Dropout | Small-to-mid scale training; often 0 for LLM pretraining | Slower convergence, cheap per-step |
| Data Augmentation | Small or narrow datasets; less used at LLM pretraining scale | Extra data pipeline complexity |
| Label Smoothing | Classification tasks with visible overconfidence | Essentially free |

These aren't mutually exclusive — a typical vision model might combine all five, while a large language model pretraining run often keeps only Weight Decay and Early-Stopping-style checkpoint selection, leaving Dropout and Data Augmentation off entirely once data scale itself is doing most of the regularizing work.

## Further reading

Nitish Srivastava et al. introduced Dropout in *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* (2014). Label smoothing was introduced alongside the Inception architecture in Christian Szegedy et al.'s *Rethinking the Inception Architecture for Computer Vision* (2016).