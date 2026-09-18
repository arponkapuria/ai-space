# Transfer Learning

## Motivation

[Self-Supervised Learning](/topic/self-supervised-learning), [Autoencoders](/topic/autoencoders), and ordinary supervised pretraining all end with the same thing: a trained model holding genuinely useful learned representations. Transfer learning is the practical follow-up question — once that representation exists, how do you actually reuse it for a *new* task, especially one with far less labeled data than the original training used? Training every new task's model completely from scratch throws away an enormous amount of already-learned general knowledge — edges and textures useful across almost any vision task, grammar and semantics useful across almost any language task — for no good reason.

## Feature Extraction vs. Fine-Tuning

**When to reach for Feature Extraction:** the new task is similar to what the model was originally trained on, and there's very little labeled data available for it.

**How it works:** freeze every pretrained weight, and train only a new, small "head" on top — often just a single linear layer — for the new task. Cheap, fast, and because almost nothing is being trained, it needs very little labeled data to avoid overfitting.

**When to reach for Fine-Tuning:** the new task differs more from the original, and there's enough labeled data to support updating more of the model without it just memorizing that smaller dataset.

**How it works:** unfreeze some or all of the pretrained weights and continue training on the new task's data — almost always with a much smaller [learning rate](/topic/learning-rate-scheduling) than the original training used, since large updates this late risk erasing what the model already knows rather than adapting it.

A common middle ground: freeze the early layers (which tend to capture generic, broadly reusable features like edges and textures) and only fine-tune the later layers (which capture more task-specific, abstract patterns). This gets most of fine-tuning's adaptability at closer to feature extraction's cost.

| | Data available | Task similarity | Approach |
|---|---|---|---|
| Case 1 | Small | Similar to original | Feature Extraction |
| Case 2 | Small | Different from original | Feature Extraction, or fine-tune only the last few layers |
| Case 3 | Large | Similar to original | Fine-tune the whole model |
| Case 4 | Large | Different from original | Fine-tune the whole model, or consider training from scratch |

## The risk: catastrophic forgetting

Fine-tuning too aggressively — too high a learning rate, too many epochs on a small new dataset — can cause **catastrophic forgetting**: the model overwrites what it originally learned faster than it learns the new task, sometimes ending up worse overall than if it had never been fine-tuned at all. This is a large part of why fine-tuning uses a small learning rate by default, and it's a risk that shows up again, in a more specific form, when large language models are adapted through [Instruction Tuning](/topic/instruction-tuning).

## Where this actually shows up in practice

The original success story is a CNN pretrained on ImageNet, with a new head swapped on for a specialized task — medical imaging, satellite imagery — where labeled data is scarce but general visual features transfer well. In NLP, this same pattern is the entire foundation of how [BERT](/topic/bert) and [GPT-1](/topic/gpt-1) get used: pretrain once on a huge unlabeled corpus, then fine-tune (or lightly adapt) for each specific downstream task. Modern large language model adaptation takes this further still — [PEFT](/topic/peft) methods like LoRA are a parameter-efficient evolution of the same core idea, updating a small fraction of the model instead of all of it, specifically to get fine-tuning's adaptability while avoiding most of its cost and catastrophic-forgetting risk.

## Further reading

Jason Yosinski et al.'s *How Transferable Are Features in Deep Neural Networks?* (2014) is the paper most directly credited with rigorously answering the feature-extraction-vs-fine-tuning question, measuring layer by layer how well CNN features transfer to new tasks.