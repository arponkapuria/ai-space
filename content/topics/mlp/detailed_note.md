# Multilayer Perceptron (MLP)

## Motivation

A single [Perceptron](/topic/perceptron) can only draw one straight line through the data — it's a linear classifier, which is why it can't solve something as simple as XOR. The obvious fix is to stack perceptrons: feed one layer's output into another. But stack plain perceptrons and you gain nothing — a straight line of straight lines is still a straight line, no matter how many you chain. What actually unlocks the extra layers is adding a **nonlinear** function at each step. That combination — layers, each doing a linear transform followed by a nonlinearity — is the Multilayer Perceptron, the first architecture that can learn genuinely curved decision boundaries. Everything commonly called a "neural network" is built on this idea.

## Architecture

An MLP is organized into layers of units ("neurons"), each doing the same basic thing a perceptron does — weighted sum plus bias — but followed by a smooth [activation function](/topic/activation-functions) instead of a hard step:

<div align="center">
  <img src="/assets/mlp.png" alt="Multi Layer Perceptron" width="500" />
</div>

- **Input layer** — not really a computing layer, just the raw feature vector $\mathbf{x}$.
- **Hidden layer(s)** — one or more layers between input and output. Each unit takes the *previous* layer's output, computes a weighted sum, and applies a nonlinear activation. "Deep" learning gets its name from stacking several of these.
- **Output layer** — produces the final prediction. Its activation depends on the task: none for regression, sigmoid for binary classification, softmax for multi-class.

For one hidden layer, the full computation is:

$$
\mathbf{h} = \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1), \qquad \hat{y} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2
$$

- $\mathbf{W}_1, \mathbf{b}_1$ — weights and bias mapping inputs to the hidden layer. Every row of $\mathbf{W}_1$ is one hidden unit's own weight vector.
- $\sigma$ — the nonlinear activation, applied elementwise (sigmoid or tanh historically; almost always [ReLU](/topic/activation-functions) or a variant today).
- $\mathbf{W}_2, \mathbf{b}_2$ — maps the hidden layer to the final prediction.

Every unit connects to every unit in the layer before it, which is why this is also called a **fully connected** or **feedforward** network.

## Why the nonlinearity is the whole point

<details>
<summary>Math: why stacking linear layers alone changes nothing</summary>

If $\sigma$ were the identity function instead of a nonlinearity, the two-layer computation above collapses:

$$
\hat{y} = \mathbf{W}_2(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (\mathbf{W}_2 \mathbf{W}_1)\mathbf{x} + (\mathbf{W}_2 \mathbf{b}_1 + \mathbf{b}_2)
$$

$\mathbf{W}_2 \mathbf{W}_1$ is just another matrix, and $\mathbf{W}_2 \mathbf{b}_1 + \mathbf{b}_2$ is just another bias — so the whole thing is mathematically identical to a single linear layer, no matter how many layers you stacked or how wide they are.

</details>

With a real nonlinearity, the **Universal Approximation Theorem** (Cybenko 1989; Hornik 1991) guarantees something stronger: a network with one hidden layer, given *enough* units, can approximate any continuous function to arbitrary precision. That's a big reason MLPs generated so much early excitement.

The theorem only promises the right weights *exist* — not that gradient descent will find them, or how many units "enough" means in practice. Going *deeper* (more layers, each moderate in size) tends to represent complex functions far more efficiently than going *wider* (one huge layer), which is the practical reason deep networks won out over the theorem's own single-layer construction.

## Training

An MLP trains like any [supervised learning](/topic/supervised-learning) model: define a [loss function](/topic/loss-functions) comparing predictions to labels, then use [Gradient Descent](/topic/gradient-descent) to adjust every weight. The new problem multiple layers add is *how* to compute the gradient for a weight buried several layers deep — that's what [Backpropagation](/topic/backpropagation) solves, using the chain rule to push the error signal backward. This is also why the perceptron's hard step function had to go: backprop needs a derivative at every layer, and a step function's derivative is zero almost everywhere. Swapping it for a smooth activation is what made training a multi-layer network possible at all.

## Design choices

Two knobs set an MLP's capacity, and they trade off differently:

- **Width** (units per layer) — more units let a layer represent more patterns, at roughly linear cost in parameters and compute.
- **Depth** (number of layers) — more layers let the network build up abstraction, composing simple patterns into complex ones. But very deep plain MLPs are harder to train — gradients can shrink or explode as they flow through many layers (see [Gradient Instability](/topic/gradient-instability)), which is why [weight initialization](/topic/weight-initialization) and [normalization techniques](/topic/normalization-techniques) matter more as depth grows.

Both directions raise the risk of overfitting on limited data — that's where [Regularization](/topic/regularization) comes in.

## A recent alternative: Kolmogorov-Arnold Networks

MLPs put the learnable part on the **edges** (the weights) and fix the nonlinearity at the **nodes** (a hardcoded activation like ReLU, applied everywhere). **Kolmogorov-Arnold Networks (KANs)**, proposed in 2024, flip that: the nonlinear functions live on the edges and are learned (usually as splines), while nodes just sum their inputs. The appeal is interpretability and sometimes better accuracy-per-parameter on small scientific problems, since you can inspect the learned edge functions directly. The trade-off: KANs train much slower and haven't shown they can scale to typical deep learning sizes — a promising research direction and a niche tool, not yet a replacement for the MLP.

## Where this leaves things

A plain MLP treats its input as an unstructured flat vector — no built-in sense that "nearby pixels are related" or "word order matters." That's why architectures with built-in structural assumptions overtook plain MLPs in their domains: convolutional networks for images, attention-based networks for sequences. What survives is the fully connected layer itself — still the workhorse *inside* nearly every modern architecture, including the feedforward sub-layers inside a Transformer block. The MLP didn't become obsolete; its role shifted from "the whole model" to "the component every other architecture still relies on."

## Further reading

The algorithm for training multilayer networks — backpropagation — was popularized by David Rumelhart, Geoffrey Hinton, and Ronald Williams in *Learning Representations by Back-propagating Errors* (1986), though the underlying idea goes back further. The theoretical guarantee that one hidden layer suffices in principle comes from George Cybenko's *Approximation by Superpositions of a Sigmoidal Function* (1989) and Kurt Hornik's *Approximation Capabilities of Multilayer Feedforward Networks* (1991).