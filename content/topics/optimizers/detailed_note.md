# Optimizers

## Motivation

[Backpropagation](/topic/backpropagation) tells you the gradient — which direction each weight should move to reduce the loss. But knowing the direction isn't the same as knowing the best way to move. Plain [Gradient Descent](/topic/gradient-descent) takes the same-sized step for every weight, every time, based only on the current gradient — it zig-zags in some situations, crawls to a stall in others, and treats every weight identically even though some clearly need bigger adjustments than others. Optimizers decide *how* to turn a gradient into an actual update, smarter than "just subtract the gradient" — and a good one can be the difference between training in hours versus never converging.

## Plain gradient descent, and where it struggles

Picture rolling a ball down a hill toward the lowest point. Each step needs a gradient, and there are three ways to get one: **batch gradient descent** uses the entire dataset (exact, but expensive per step), pure **stochastic gradient descent** uses a single random example (cheap, but noisy), and **mini-batch gradient descent** — a small batch of maybe 32 to a few thousand examples — splits the difference, and is what's almost universally used in practice. This is also why "SGD" colloquially means the mini-batch version in nearly every real deep learning context, not the literal one-example-at-a-time version. See [Gradient Descent](/topic/gradient-descent) for the full comparison.

Whichever variant is used, the direction each step takes is noisy relative to the true gradient. That noise causes two recurring problems. In a **narrow valley** — steep on one side, gentle on the other — a ball bounces back and forth across the steep walls instead of heading straight for the bottom. On a **flat stretch**, the gradient is tiny, so the steps are tiny too, and training crawls. Both come from the same cause: gradient descent only looks at *right now* — no memory of past direction, no sense of which weights need bigger or smaller steps.

## Momentum and Nesterov: smoothing out the direction

**Momentum** gives the ball actual weight — like a heavy ball that doesn't instantly change direction every time the slope wiggles. It keeps a running average of recent gradient directions and moves along that instead of just the current gradient. Directions that keep repeating reinforce each other and speed up; directions that keep flipping back and forth cancel out.

<details>
<summary>Math: the momentum update</summary>

$$
v \leftarrow \beta v + (1-\beta) \nabla L, \qquad w \leftarrow w - \eta v
$$

$v$ is the running average of past gradients, $\beta$ controls how much past direction carries forward (typically ~0.9), and $\nabla L$ is the current gradient.

</details>

**Nesterov Accelerated Gradient (NAG)** improves this slightly: instead of computing the gradient from where the weight currently sits, it first steps in the direction momentum is already carrying it, *then* checks the gradient from that lookahead spot. It's peeking ahead to where momentum is about to carry you and correcting early, rather than reacting after the fact — a bit faster and more stable than plain Momentum, for almost the same cost.

## Adagrad and RMSProp: adapting the step size

Momentum and Nesterov fix *direction*. A separate problem is *step size* — some weights sit where gradients are consistently small and could take bigger steps safely; others need smaller, more careful ones. One shared learning rate ignores this.

**Adagrad** keeps a running *sum* of every squared gradient a weight has ever received, and shrinks that weight's step as the sum grows — useful for rarely-updated weights (like an embedding for a rare word), which keep taking full-sized steps. The flaw: a sum only ever grows, so eventually every weight's step shrinks toward zero and training stalls, even if there's more to learn.

**RMSProp** fixes this by using a moving *average* of recent squared gradients instead of an ever-growing sum — old history fades out, so the step size can recover instead of only ever shrinking. Like adjusting your stride to how rough the ground has *recently* been under each foot, not permanently shortening it the first time it was ever rough.

## Adam and AdamW

**Adam** combines both ideas: a running average of the gradient's *direction* (momentum) and a running average of its recent *size* (RMSProp), used together for each weight's update. That combination — remember where you've been heading, adjust step size per weight based on recent volatility — is why Adam became the default choice for training almost any network.

<details>
<summary>Math: the Adam update</summary>

$$
m \leftarrow \beta_1 m + (1-\beta_1)\nabla L, \qquad s \leftarrow \beta_2 s + (1-\beta_2)(\nabla L)^2
$$

$$
w \leftarrow w - \eta \frac{m}{\sqrt{s} + \epsilon}
$$

$m$ tracks direction, $s$ tracks recent gradient size, $\epsilon$ just avoids dividing by zero. A small bias correction is applied to both early in training, since they start at zero.

</details>

[Regularization](/topic/regularization) often adds weight decay — a small pull toward zero each step. Plain Adam applies this *before* its adaptive scaling, which distorts how much decay each weight actually gets. **AdamW** applies decay separately, after the adaptive step, fixing this — and is now the default for training almost every large model, LLMs included.

## At the scale of large language models

Training with tens or hundreds of billions of parameters adds problems that only appear at that size — Adam's per-weight memory cost alone can outweigh the model's own weights, and very large batch sizes stop being stable by default. **Adafactor** (memory-efficient), **LAMB** (stable at massive batch sizes), and newer entrants like **Muon** and **Adam-mini** exist to address exactly this. They're covered under [Pretraining](/topic/pretraining) in LLM Engineering, where these scale problems actually show up.

## Picking one in practice

| Optimizer | Good for | Trade-off |
|---|---|---|
| **SGD + Momentum** | Vision models, when final quality matters more than training speed | More manual tuning; sometimes generalizes slightly better once tuned |
| **Adam / AdamW** | Almost everything else — the default, especially for Transformers and LLMs | Faster convergence, less tuning; sometimes a touch worse generalization than tuned SGD |

Starting with Adam or AdamW is the standard default across nearly all of deep learning today; tuned SGD with momentum is mostly a later choice for specific vision architectures where it's known to help.

## Further reading

The momentum idea traces back to Boris Polyak's *Some Methods of Speeding Up the Convergence of Iteration Methods* (1964), with Yurii Nesterov's accelerated variant following in 1983. John Duchi, Elad Hazan, and Yoram Singer introduced Adagrad in *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization* (2011). RMSProp comes from an unpublished 2012 lecture by Geoffrey Hinton and Tijmen Tieleman. Diederik Kingma and Jimmy Ba introduced Adam in *Adam: A Method for Stochastic Optimization* (2015). Ilya Loshchilov and Frank Hutter introduced AdamW in *Decoupled Weight Decay Regularization* (2017).