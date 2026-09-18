# Weight Averaging (SWA & EMA)

## Motivation

Training doesn't converge to one single, perfect point — an optimizer like SGD or Adam keeps taking steps even after the model has mostly plateaued, wandering around within a region of good solutions rather than settling exactly on one. Whatever weights you happen to have when training stops are just wherever that wandering ended up, not necessarily the best point along the way. This isn't the same problem [Generalization Techniques](/topic/generalization-techniques) like Dropout or Data Augmentation solve — those change what the model *learns* during training. Weight averaging changes nothing about training itself; it's a nearly-free adjustment to *which* weights get kept and used afterward.

## The core idea

Picture asking five friends to each estimate a number, then averaging their answers — the average tends to land closer to right than trusting whichever one friend happened to answer last. Weight averaging applies the same idea to training: instead of keeping only the final set of weights, blend together several sets from along the way.

## Stochastic Weight Averaging (SWA)

**When to reach for it:** mostly vision model training and research or competition settings, where squeezing out an extra 1–2% accuracy is worth some modest extra bookkeeping.

**How it works:** near the end of training, save the weights every so often — say, once per epoch for the last several epochs — and once training finishes, average all of those saved copies together, equally. That averaged version is what actually gets used going forward, not the very last training step. It tends to land in a flatter, more stable region than any single snapshot, which generalizes a bit better.

<details>
<summary>Math and implementation details</summary>

$$
w_{\text{SWA}} = \frac{1}{n}\sum_{i=1}^{n} w_i
$$

where each $w_i$ is a saved snapshot of the weights. SWA is normally paired with a specific learning rate schedule during this final phase — a constant or cyclical rate, rather than one still decaying toward zero — so the weights keep exploring the region around a good solution instead of converging tightly onto one single point before the snapshots are taken.

</details>

It's a genuinely useful trick — commonly cited gains are around 1–1.5% accuracy on ImageNet-scale vision benchmarks, and it ships built into PyTorch — but it's fair to say it's more of a known trick reached for in vision research and competitions (Kaggle especially) than something used by default in most production training pipelines.

## Exponential Moving Average (EMA)

**When to reach for it:** routinely, in diffusion models, GAN training, and self-supervised setups — it's cheap enough that there's rarely a reason not to.

**How it works:** instead of averaging a few snapshots at the end, EMA maintains a second, separate copy of the weights that quietly follows the real one throughout *all* of training. After every single step, this "shadow" copy doesn't jump straight to match the real weights — it only moves part of the way there, nudged gently toward wherever the real weights currently are. Repeated over thousands of steps, the shadow copy ends up as a smoothed-out version of the whole training trajectory, with recent weights influencing it more than very old ones, but nothing thrown away entirely.

<details>
<summary>Math and implementation details</summary>

$$
w_{\text{EMA}} \leftarrow \alpha \, w_{\text{EMA}} + (1-\alpha) \, w_{\text{current}}
$$

$\alpha$ (commonly 0.999 or 0.9999) controls how slowly the shadow copy moves — closer to 1 means it changes more slowly and smooths out more noise, at the cost of lagging further behind the actual current weights.

</details>

This shadow copy is what actually gets used in several widely-used setups: diffusion models like Stable Diffusion generate images from the shadow copy rather than the raw weights being actively trained, because it noticeably improves output quality; GAN training keeps a shadow copy of the generator for the same reason, since raw GAN training is notoriously jumpy; and self-supervised methods like BYOL and DINO use a shadow copy as a stable "teacher" for the actively-training network to learn from. It's cheap — one extra copy of the weights, nudged a little every step — which is a big part of why it's used so much more widely than SWA in practice.

## Picking one in practice

| | SWA | EMA |
|---|---|---|
| When averaging happens | Once, at the end of training | Continuously, throughout training |
| Setup needed | A modified learning rate schedule for the final phase | None — just an extra weight buffer |
| Common in | Vision research, competitions | Diffusion models, GANs, self-supervised learning |

## Further reading

Boris Polyak and Anatoli Juditsky's *Acceleration of Stochastic Approximation by Averaging* (1992) established the general idea of averaging optimizer iterates, which both SWA and EMA build on. Pavel Izmailov et al. introduced Stochastic Weight Averaging specifically for deep networks in *Averaging Weights Leads to Wider Optima and Better Generalization* (2018).