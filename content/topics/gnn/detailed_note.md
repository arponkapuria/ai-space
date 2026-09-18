# GNN (Graph Neural Networks)

## Motivation

Everything covered so far assumes data with a regular, fixed shape — a flat feature vector for an [MLP](/topic/mlp), a grid of pixels for a CNN, an ordered sequence for an RNN or Transformer. A lot of real data doesn't look like that at all: a social network (people as nodes, friendships as edges), a molecule (atoms as nodes, bonds as edges), a citation network, a road network. A graph has no fixed number of neighbors per node and no natural ordering the way a sequence does, so none of the architectures above can consume it directly. GNNs are built specifically to learn from this irregular, connected structure instead of forcing it into a shape it doesn't naturally have.

## The core idea: message passing

**Concrete example:** predicting whether someone will like a product, given their position in a social network. Their own features (age, past purchases) matter, but so does context from their neighbors — if most of their friends bought the product, that's real signal too. A GNN's central mechanism, **message passing**, formalizes exactly this: each node updates its own representation by aggregating information from its neighbors, repeated over several rounds. After $k$ rounds (layers), a node's representation has absorbed information from everything within $k$ hops in the graph — no longer just its own features, but a summary of its surrounding neighborhood, $k$ steps out.

The general recipe, repeated at every layer: for each node, gather the current representations of its neighbors, aggregate them somehow (sum, mean, or something learned), and combine that aggregated neighbor information with the node's own current representation to produce its updated one. Where the specific GNN variants below differ is entirely in *how* that aggregation step works.

## GCN (Graph Convolutional Network)

**When to reach for it:** the default starting point for node or graph classification, on a graph whose full structure is known upfront.

**How it works:** aggregate a neighbor's features as a normalized, weighted average — normalized by node degree, so a node with unusually many neighbors doesn't dominate the average purely by having more of them. This plays a similar role to a CNN's convolution: averaging over a local neighborhood — except the "neighborhood" here is defined by the graph's actual edges instead of a fixed pixel grid.

**The limitation:** GCN's normalization depends on knowing the whole graph structure at training time, which means it doesn't naturally generalize to brand-new nodes that weren't present during training — it's **transductive**, tied to the specific graph it was trained on.

## GraphSAGE

**When to reach for it:** the graph grows over time (new nodes appear after training) or is too large to process all at once — a recommendation system's constantly-changing user/item graph is the canonical example.

**How it works:** instead of using a node's entire neighborhood and requiring the full graph upfront, GraphSAGE **sa**mples a fixed-size subset of each node's neighbors and **a**ggre**g**at**e**s over just that sample, using a learned aggregation function rather than a structure-specific formula. Because it learns a general function instead of baking in the specific graph it was trained on, it can handle entirely new, unseen nodes at inference time — this is called **inductive**, and it's the single biggest practical difference from GCN.

## GAT (Graph Attention Network)

**When to reach for it:** not every neighbor should count equally — in a citation network, some cited papers are far more relevant to a paper's actual topic than others, and a fixed average treats them all the same.

**How it works:** instead of a fixed degree-normalized average (GCN) or a fixed-size sample (GraphSAGE), GAT uses an [attention mechanism](/topic/attention-mechanism) to learn a weight for each neighbor dynamically, based on how relevant that specific neighbor's features actually are to the node being updated. Some neighbors end up contributing much more than others — learned from data, rather than fixed by node degree or sample size.

## Picking one in practice

| Method | Aggregation | Generalizes to new nodes? | Best for |
|---|---|---|---|
| GCN | Fixed, degree-normalized average | No (transductive) | A fixed, fully-known graph |
| GraphSAGE | Sampled neighbors, learned aggregation function | Yes (inductive) | Large or growing graphs, new nodes over time |
| GAT | Learned attention weight per neighbor | Yes, typically | Graphs where neighbor relevance varies a lot |

## Where this actually shows up in practice

- **Recommendation systems.** Pinterest's PinSage is GraphSAGE applied at production scale, over a graph of pins and boards, to power recommendations.
- **Drug discovery.** Molecules are naturally graphs (atoms as nodes, bonds as edges), and GNNs are widely used to predict molecular properties directly from that structure.
- **Fraud detection.** Transaction graphs (accounts as nodes, transactions as edges) are a natural fit for spotting unusual connectivity patterns.
- **Social network analysis and knowledge graphs.** Friend recommendation, influence prediction, and reasoning over relationships in a knowledge graph all build on the same message-passing foundation.

A GNN's final node representations are still just another instance of [Representation Learning](/topic/representation-learning)'s core goal — a compact, reusable encoding — just produced by a mechanism built specifically for graph structure instead of grids or sequences.

## Further reading

Thomas Kipf and Max Welling introduced GCN in *Semi-Supervised Classification with Graph Convolutional Networks* (2017). William Hamilton, Rex Ying, and Jure Leskovec introduced GraphSAGE in *Inductive Representation Learning on Large Graphs* (2017). Petar Veličković et al. introduced GAT in *Graph Attention Networks* (2018).