# Learning Rate Scheduling

## Motivation

The learning rate ($\eta$ in [Gradient Descent](/topic/gradient-descent)) controls how big each weight update is, and a single fixed value is always a compromise. Early in training, the model is far from any minimum, and a larger learning rate makes fast progress. Later, as the loss surface narrows near a good solution, that same large learning rate causes overshooting — bouncing around the minimum instead of settling into it. A learning rate small enough to be safe near the end wastes most of training being too cautious near the start. Learning rate scheduling fixes this by changing the learning rate over the course of training instead of keeping it fixed.

## Two ways to decide when to change it

Every schedule below falls into one of two camps. **Scheduled** approaches decide the entire learning rate curve in advance, before training starts, and follow it regardless of how training actually goes. **Reactive** approaches instead watch training as it happens and adjust based on what's observed. Scheduled approaches are simpler and dominate large-scale training, since they're predictable and easy to reproduce; reactive approaches adapt to the actual run, at the cost of being harder to predict in advance.

## Scheduled: fixed decay curves

These all answer the same question — "how should the learning rate shrink over time?" — with a different curve shape:

- **Step Decay** drops the learning rate by a fixed factor (e.g. ×0.1) every fixed number of epochs. Simple, but each drop causes a visible jump in the loss curve right at that point.
- **Exponential Decay** shrinks the learning rate by a fixed percentage every step instead, giving a smooth curve instead of Step Decay's sudden jumps.
- **Linear Decay** shrinks it in a straight line down to a target value. Simple and common, especially paired with warmup (next section) in large language model pretraining.
- **Cosine Annealing** follows a cosine curve down to near zero — slow at first, faster through the middle, tapering gently again near the end. That smooth taper tends to outperform Step Decay's abrupt drops, and it's one of the most widely used schedules today.

A well-known variant, **Cosine Annealing with Warm Restarts (SGDR)**, doesn't decay to zero just once — it periodically resets the learning rate back up and lets it decay again, repeating the cycle several times. Each reset gives the optimizer a fresh burst of exploration, useful for escaping a mediocre minimum it had started to settle into.

## Scheduled: Warmup

Right at the start of training, weights are freshly initialized and gradients can be unusually large or noisy. For adaptive optimizers like [Adam](/topic/optimizers), the running estimates of gradient direction and size haven't had time to become reliable either. Taking a full-sized step under these conditions risks a bad, hard-to-recover-from update before training has even really begun. **Warmup** starts the learning rate near zero and ramps it up gradually — over the first few hundred to a few thousand steps — giving the model, and the optimizer's internal statistics, time to stabilize before the full learning rate kicks in.

Warmup is almost never used by itself — it's the on-ramp, not the destination. A typical setup ramps the learning rate up from 0 over the first couple thousand steps, then hands off to a decay schedule for the rest of training. **Warmup followed by Cosine Annealing** is the specific combination used in most modern large-model training, including GPT and BERT-style pretraining.

## Scheduled: One-Cycle Policy

The **One-Cycle Policy** takes warmup and decay and fuses them into one deliberate shape, rather than treating them as two separate phases: the learning rate rises from a low starting point up to a high peak over roughly the first half of training, then decays back down — often ending even lower than where it started. It looks similar to Warmup + Cosine Annealing on a graph, but the intent differs: the *peak* itself is the point, not just a ramp-up target. Pushing the learning rate unusually high in the middle of training acts like a regularizer, forcing the model through a wider part of the loss landscape rather than letting it settle too quickly into a narrow, possibly mediocre solution. In practice, this often reaches a good result in noticeably fewer total steps than a plain decay schedule.

## Reactive: ReduceLROnPlateau

Every approach so far decides its curve before training starts. **ReduceLROnPlateau** does the opposite: it watches a validation metric during training, and only lowers the learning rate once that metric stops improving for a set number of epochs. It adapts to how this specific run is actually going, rather than following a fixed guess made in advance — a common default outside large-scale Transformer training, where fixed warmup-plus-decay schedules tend to dominate instead.

## Picking one in practice

| Setting | Typical choice |
|---|---|
| Training a Transformer / LLM from scratch | Warmup, then Cosine Annealing (or Linear Decay) |
| Fine-tuning a pretrained model | Short warmup, then Linear or Cosine Decay |
| Classic CNN training (vision, tabular) | Step Decay, or ReduceLROnPlateau |
| Training with limited compute budget, need fast results | One-Cycle Policy |

## Further reading

Ilya Loshchilov and Frank Hutter introduced Cosine Annealing with Warm Restarts in *SGDR: Stochastic Gradient Descent with Warm Restarts* (2016). Leslie Smith introduced the One-Cycle Policy in *A Disciplined Approach to Neural Network Hyper-Parameters* (2018). Priya Goyal et al.'s *Accurate, Large Minibatch SGD* (2017) is a widely cited paper demonstrating why warmup matters for training stability at large batch sizes.