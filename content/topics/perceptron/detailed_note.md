# Perceptron

## Motivation

[Logistic Regression](/topic/logistic-regression) learns a straight-line boundary by minimizing a smooth loss with [Gradient Descent](/topic/gradient-descent). But that setup — a smooth loss, gradients, calculus — came later. The question came first, in a cruder form: could a machine learn the way a brain's neurons seemed to, by taking in signals, weighing them, and firing or not firing? Frank Rosenblatt's 1958 Perceptron was the first algorithm to answer that with something you could actually train on data. It's the direct ancestor of every neural network since — strip away a modern deep net's layers and training tricks, and what's left at the bottom is still, structurally, a perceptron.

## The model

<div align="center">
  <img src="/assets/perceptron.png" alt="Multi Layer Perceptron" width="500" />
</div>

A perceptron takes a list of inputs, adds them up with weights, and fires 1 or 0 depending on whether that sum clears a threshold:

$$
\hat{y} = \begin{cases} 1 & \text{if } \mathbf{w} \cdot \mathbf{x} + b > 0 \\ 0 & \text{otherwise} \end{cases}
$$

- $\mathbf{x}$ — the input features.
- $\mathbf{w}$ — the weights, one per feature. Each one controls how much that feature pushes the decision toward 1 or 0.
- $b$ — the bias, which shifts the threshold. Without it, the boundary would be stuck passing through the origin.
- The **step function** is what makes this a classifier and not just linear regression — it turns a number into a hard yes/no.

Geometrically, $\mathbf{w} \cdot \mathbf{x} + b = 0$ is a straight line (or a flat plane in higher dimensions) cutting the input space in two. That's the same picture as logistic regression's boundary — what differs is how the line gets fit, and how confident the output is allowed to be.

## How it learns

Instead of gradient descent on a smooth loss, the original perceptron updates weights straight from its mistakes, one example at a time:

$$
\mathbf{w} \leftarrow \mathbf{w} + \eta (y - \hat{y})\, \mathbf{x}, \qquad b \leftarrow b + \eta (y - \hat{y})
$$

$y$ is the true label, $\hat{y}$ the current guess, $\eta$ the learning rate. What this actually does:

- Guess already correct? $(y - \hat{y}) = 0$, nothing changes.
- Guessed 0, should've been 1? Nudge the weights toward $\mathbf{x}$, so next time inputs like this one push the sum higher.
- Guessed 1, should've been 0? Nudge away from $\mathbf{x}$, pushing the sum lower.

Training loops over the data, fixing one mistake at a time, until everything's classified correctly or a max number of passes is hit.

<details>
<summary>Math: the Perceptron Convergence Theorem</summary>

If the data is **linearly separable** — some line perfectly divides the two classes — the perceptron is *guaranteed* to find that line in a finite number of updates. Formally, if a unit vector $\mathbf{w}^*$ and margin $\gamma > 0$ exist such that $y_i(\mathbf{w}^* \cdot \mathbf{x}_i) \geq \gamma$ for every example, and every input satisfies $\lVert \mathbf{x}_i \rVert \leq R$, the number of mistakes before convergence is bounded by:

$$
\text{mistakes} \leq \left(\frac{R}{\gamma}\right)^2
$$

Smaller margin (barely separable classes) or larger inputs mean more corrections needed. This was a big deal in 1958 — a learning algorithm with a provable convergence guarantee, not just a heuristic that seemed to work. It's a large part of why the perceptron caused so much early excitement.

The catch: this says nothing about data that *isn't* linearly separable. There, the algorithm never converges — it just keeps correcting forever.

</details>

## The limitation that mattered

A perceptron can only learn a straight-line boundary. Any dataset that isn't linearly separable is simply unsolvable for it, no matter how long you train.

The classic example is **XOR**: a function of two binary inputs that outputs 1 when exactly one of them is 1. Plot the four points and no single straight line separates the 1s from the 0s. Marvin Minsky and Seymour Papert proved this limitation rigorously in their 1969 book *Perceptrons*. That proof — simple, widely read, showing the model everyone was excited about couldn't even learn XOR — is credited with pulling funding and attention away from neural network research for over a decade, now called the first **AI winter**.

The fix wasn't a better learning rule for one perceptron — it was stacking several together. A layer of perceptrons feeding into another can carve out curved boundaries a single one never could. That's the idea behind the [Multilayer Perceptron (MLP)](/topic/mlp). One catch: the perceptron's learning rule needs a hard step function, and stacking hard steps gives no usable gradient to push backward through multiple layers. Solving that is exactly why [Backpropagation](/topic/backpropagation) and smooth [Activation Functions](/topic/activation-functions) came later.

## Perceptron vs. logistic regression

Both learn the same kind of boundary, but differ in exactly the ways that mattered for what came next:

| | Perceptron | Logistic Regression |
|---|---|---|
| Output | Hard 0/1 | Probability in $(0, 1)$ |
| Learning rule | Fix mistakes one at a time | Gradient descent on a smooth loss |
| Guarantee | Converges only if data is separable; loops forever otherwise | Always converges — the loss is convex either way |
| Confidence | None | Yes — a probability tells you how sure |

That hard step is exactly what breaks the chain for building deeper networks: there's no gradient to pass through a non-differentiable threshold.

## Where this leaves things

A single perceptron is rarely used today — logistic regression's probability output and cleaner guarantees make it the better default wherever one straight line is enough. What survives is the *structure*: a weighted sum plus a bias, passed through a nonlinearity, is still exactly what one unit in a modern neural network computes. The [MLP](/topic/mlp), [Activation Functions](/topic/activation-functions), and [Backpropagation](/topic/backpropagation) are all about fixing what one perceptron can't do — while keeping this same basic unit as the building block.

## Further reading

Frank Rosenblatt introduced the perceptron and its learning rule in *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain* (1958), building on Warren McCulloch and Walter Pitts' earlier 1943 mathematical model of a neuron. Marvin Minsky and Seymour Papert's *Perceptrons* (1969) proved the linear-separability limitation and is widely cited as the trigger for the first AI winter.