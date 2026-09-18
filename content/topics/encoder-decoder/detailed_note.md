# Encoder-Decoder

## Motivation

[Representation Learning](/topic/representation-learning) sets the goal: turn raw data into a compact, useful encoding. But that goal doesn't say what a network built to do this should actually look like. The most natural shape turns out to be splitting the network into two halves: one half compresses the input down into a representation, and a second half expands that representation back out into whatever output is actually needed. This split — **encoder, then decoder** — is one of the most reused architectural patterns in deep learning, showing up in largely the same shape across otherwise very different problems.

## The core idea

The **encoder** takes the raw input $x$ and compresses it into a representation $z$: $z = f(x)$. The **decoder** takes $z$ and produces the desired output $y$: $y = g(z)$. The two halves are almost always trained together, end to end, so the whole pipeline learns to work as one system.

**Why the split matters:** everything the decoder is allowed to know about the input has to pass through $z$ first — $z$ is the only channel connecting the two halves. This forces the encoder to keep whatever's actually necessary for producing a good output, and forces the decoder to work from that compressed summary alone, rather than having direct access to the raw input. $z$ is doing real work here, not just sitting between two halves as a formality.

## The same shape, different jobs

What stays constant across every encoder-decoder architecture is the split itself. What changes is what $x$ and $y$ actually are, what the encoder and decoder are built from internally, and what shape $z$ takes:

| Architecture | $x \to y$ | Built from | $z$ |
|---|---|---|---|
| [Autoencoder](/topic/autoencoders) | Data → the same data, reconstructed | MLP / CNN | A single fixed-size vector |
| Seq2Seq (original) | A sentence → its translation | RNN / LSTM | A single fixed vector (the "context vector") |
| [Transformer](/topic/transformer) (encoder-decoder variant) | A sentence → its translation | Self-attention blocks | A full sequence of vectors, not squeezed to one |
| U-Net | An image → a segmentation map | CNN | Multi-scale feature maps, passed directly to the decoder via skip connections |
| VAE | Data → the same data, reconstructed (and new samples) | MLP / CNN | A probability distribution, not a fixed point |

## The bottleneck problem — and how later architectures fixed it

The original sequence-to-sequence architecture squeezed an entire input sentence into one single fixed-size context vector, regardless of how long the sentence was. For short sentences this was fine; for long ones, cramming everything into one fixed-size vector meant real information loss — the decoder simply couldn't recover detail that never made it into $z$ in the first place. This specific limitation is exactly what motivated the [Attention Mechanism](/topic/attention-mechanism): instead of forcing the decoder to work from one compressed vector, attention lets it look back at *all* of the encoder's intermediate states directly, at every decoding step.

U-Net solves a related but distinct version of this problem for images: a single bottleneck vector loses the precise spatial detail (exactly *where* things are in the image) needed for tasks like segmentation. Its fix is **skip connections** — passing the encoder's intermediate feature maps directly across to the matching decoder layer, bypassing the single bottleneck entirely for the fine-grained detail, while the bottleneck itself still carries the abstract, high-level summary.

## Encoder-decoder vs. decoder-only

Not every modern architecture keeps both halves. Most current large language models (the GPT family, for instance) are **decoder-only** — there's no separate encoder step at all. As a rough guide: when the task deeply restructures one input into a different kind of output with a clear source to stay grounded to — translation, summarization, segmentation — the explicit encoder-decoder split still tends to help. When the task is mostly open-ended generation continuing from a prompt, decoder-only has proven simpler to train and scale. This distinction is covered in more depth in [Transformer](/topic/transformer).

## Further reading

Kyunghyun Cho et al. introduced the RNN encoder-decoder architecture in *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation* (2014), with Ilya Sutskever, Oriol Vinyals, and Quoc Le popularizing it for general sequence-to-sequence tasks in *Sequence to Sequence Learning with Neural Networks* (2014).