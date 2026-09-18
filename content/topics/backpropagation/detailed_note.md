# Backpropagation

## Motivation

An [MLP](/topic/mlp) is trained by adjusting every weight to reduce the loss, using [Gradient Descent](/topic/gradient-descent). That means you need the gradient — how much the loss changes if you nudge one specific weight — for *every single weight in the network*, including ones buried several layers deep. A deep weight doesn't touch the loss directly; it only affects the loss through a long chain of everything downstream of it. Before an efficient method existed, computing each weight's gradient one at a time (nudge it slightly, rerun the whole network, see how much the loss moved) was so slow it made anything beyond tiny networks impractical. Backpropagation is the algorithm that fixed this: one pass forward through the network, then one pass backward, and every weight's gradient falls out along the way — computed once, not one weight at a time.

## The idea, in plain terms

Think of the network as an assembly line, where each layer hands its output to the next. If the final product comes out wrong, you'd trace the blame backward: the last station gets checked first, then whatever fed into it, then whatever fed into that, all the way back to the start. Backpropagation does exactly this with numbers — it starts at the loss (how wrong the final output was) and works backward, layer by layer, figuring out how much each weight contributed to that error.

The tool that makes this work is the **chain rule** from calculus: if $A$ affects $B$, and $B$ affects $C$, then how much a small change in $A$ moves $C$ is just the effect of $A$ on $B$, multiplied by the effect of $B$ on $C$. Backpropagation is that idea applied over and over, one layer at a time, all the way from the output back to the very first weight.

## The pattern, in general

Numbers make this far easier to follow than symbols alone. Picture the smallest possible network: one input goes in, passes through one hidden unit, and comes out as one final prediction — two weights, total. The mechanics work exactly the same way in a network with a million weights, just repeated more times. The full step-by-step walkthrough is below, but the short version is: start at the loss, and repeatedly ask "if this one thing wiggles a little, how much does the very next thing wiggle?" — multiplying each answer into the next as you move backward.

For a network with several layers, the gradient of the loss with respect to a weight $w^{(l)}$ in layer $l$ written out in full is:

$$
\frac{\partial L}{\partial w^{(l)}} = \frac{\partial L}{\partial a^{(L)}} \cdot \frac{\partial a^{(L)}}{\partial a^{(L-1)}} \cdots \frac{\partial a^{(l+1)}}{\partial a^{(l)}} \cdot \frac{\partial a^{(l)}}{\partial w^{(l)}}
$$

where $a^{(k)}$ is layer $k$'s output. This looks dense, but it's exactly the chain of nudge-and-see questions from the worked example below, just written for a layer with many weights at once instead of one connection at a time — a long multiplication of "how much did this affect the very next thing," stretching from the loss all the way back to $w^{(l)}$.

<details>
<summary>Full numeric walkthrough: forward pass, then backward pass, step by step</summary>

**Setup:** input $x = 2$, hidden weight $w_1 = 0.5$, output weight $w_2 = 1.0$. Both biases are set to 0, just to keep the numbers simple. The hidden unit uses sigmoid, a function that squashes any number down into somewhere between 0 and 1. The target value we want the network to predict is $y = 1$.

**Step 1: the forward pass.** This just means running the numbers through the network in order, left to right, and writing down every value along the way — we'll need each of them again in a moment.

- The hidden unit first multiplies the input by its weight: $z_1 = w_1 \times x = 0.5 \times 2 = 1.0$.
- Then it squashes that number through sigmoid: $h = \text{sigmoid}(1.0) = 0.731$. "Squash" just means: whatever number goes in, sigmoid always hands back something between 0 and 1.
- The output does its own multiply: $\hat{y} = w_2 \times h = 1.0 \times 0.731 = 0.731$. This is the network's final guess.
- Finally, we check how wrong that guess was: $L = \frac{1}{2}(\hat{y} - y)^2 = \frac{1}{2}(0.731 - 1)^2 = 0.036$. The bigger the gap between the guess and the target, the bigger this number gets — that's the whole point of a loss.

So: the network guessed $0.731$, the correct answer was $1$, and we now have one number, $0.036$, measuring how wrong it was. The question now is: which of the two weights should change, and in which direction, to make that number smaller?

**Step 2: the backward pass.** Before doing any of the math, it helps to say plainly what we're actually computing at every step: *"if I nudge this number up by a tiny amount, does the loss go up or down, and how fast?"* That's all a gradient is — a nudge-and-see number. We start at the loss and work backward, asking one nudge-and-see question at a time, because each answer becomes exactly what we need to ask the next question.

1. **Nudge the guess, see the loss.** If $\hat{y}$ went up a tiny bit, how much would the loss move? For this kind of loss, the answer is simply $\hat{y} - y = 0.731 - 1 = -0.269$. The negative sign just means: right now, if the guess goes up, the loss goes *down* — which makes sense, since our guess (0.731) is still below the target (1), so getting closer to 1 should help.

2. **Nudge $w_2$, and see what happens to the loss.** $w_2$ doesn't touch the loss directly — it only affects $\hat{y}$, and $\hat{y}$ is what affects the loss. So we split this into two easy questions and multiply the answers together:
   - If $w_2$ went up a tiny bit, how much would $\hat{y}$ move? Since $\hat{y} = w_2 \times h$, the answer is just $h = 0.731$.
   - Multiply that by what we already found in step 1: $-0.269 \times 0.731 = -0.197$.
   - That's the full answer for $w_2$: nudging it up a little would move the loss by about $-0.197$ times the size of the nudge.

3. **Nudge $h$, and see what happens to the loss.** Same trick, just for the hidden unit's output instead of $w_2$.
   - If $h$ went up a tiny bit, how much would $\hat{y}$ move? Since $\hat{y} = w_2 \times h$, the answer is $w_2 = 1.0$.
   - Multiply by step 1's answer: $-0.269 \times 1.0 = -0.269$.

4. **Nudge $z_1$, and see what happens to $h$.** Before $z_1$ can affect the loss at all, it first has to pass through sigmoid to become $h$. So: if $z_1$ went up a tiny bit, how much would $h$ move? For sigmoid, there's a neat shortcut — the answer is always $h \times (1 - h)$. Plugging in our numbers: $0.731 \times (1 - 0.731) = 0.731 \times 0.269 = 0.197$.
   - Multiply by step 3's answer: $-0.269 \times 0.197 = -0.053$.

5. **Nudge $w_1$, and see what happens all the way down the chain.** Last step. If $w_1$ went up a tiny bit, how much would $z_1$ move? Since $z_1 = w_1 \times x$, the answer is just $x = 2$.
   - Multiply by step 4's answer: $-0.053 \times 2 = -0.106$. That's the full gradient for $w_1$.

Notice the pattern repeating in every single step: it was always just one small, local question — *"if this one thing wiggles a little, how much does the very next thing wiggle?"* — and each answer got carried forward by multiplying it into the next question. By the time we reached $w_1$, buried furthest from the loss, we didn't need to redo any earlier work or touch $w_2$ again. We just reused the numbers already sitting there from the forward pass, and from the step right before it.

With both gradients now known — $-0.197$ for $w_2$ and $-0.106$ for $w_1$ — gradient descent would push each weight in the *opposite* direction of its gradient (a negative gradient means increasing the weight decreases the loss), moving the network one small step closer to guessing correctly next time.

</details>

## Why this is efficient

The forward pass computes and stores every intermediate value ($z_1$, $h$, $\hat{y}$, and so on). The backward pass then reuses those stored values instead of recomputing anything from scratch. This is the entire efficiency gain: one forward pass plus one backward pass gives you the gradient for *every* weight in the network, compared to the naive alternative of nudging each weight individually and rerunning the whole forward pass to see what changed — which would need one full forward pass *per weight*, and modern networks have billions of them.

This is also exactly what "autodiff" (automatic differentiation) in frameworks like PyTorch or JAX is doing under the hood — the framework builds a computational graph of every operation during the forward pass, then walks it backward automatically, applying the same chain-rule bookkeeping shown above without you writing any of the derivative math by hand.

## What can go wrong

Look back at step 4 in the example — multiplying by the activation's derivative. Chain enough of these multiplications together across many layers, and if each local derivative is consistently small (as with sigmoid, whose derivative maxes out at 0.25), the product shrinks toward zero the further back you go — the **vanishing gradient problem**. If local derivatives are consistently larger than 1 instead, the product can explode instead. This is exactly why the choice of [activation function](/topic/activation-functions) matters so much for deep networks, and it's covered in more depth under [Gradient Instability](/topic/gradient-instability).

## Where this fits

Backpropagation only computes gradients — it doesn't decide how to use them. That's the job of an optimizer: plain [Gradient Descent](/topic/gradient-descent) or one of its variants actually applies the gradients backpropagation produces to update every weight. Backprop is the "how much should this change and in which direction" calculation; gradient descent is the "okay, now actually make that change" step.

## Further reading

David Rumelhart, Geoffrey Hinton, and Ronald Williams popularized backpropagation for training neural networks in *Learning Representations by Back-propagating Errors* (1986). The underlying mathematical idea — reverse-mode automatic differentiation — is older, tracing back to Seppo Linnainmaa's 1970 thesis and Paul Werbos's 1974 dissertation, which first applied it specifically to neural networks.