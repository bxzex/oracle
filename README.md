# Oracle

A character-level transformer — real multi-head causal self-attention, layer
norm, residuals, GELU and Adam — trained in the browser tab on whatever text you
paste, with its attention shown as it reads.

Live: https://bxzex.github.io/oracle/

## The model

```
token embedding + learned positional embedding
  ├─ pre-LN → multi-head causal self-attention → residual
  └─ pre-LN → MLP (4×, GELU) → residual          × N blocks
final LN → tied output head → softmax
```

Default is 2 blocks, 3 heads, 48 embedding dimensions and a 48 character
context: about 61,000 parameters. The output head shares weights with the token
embedding, so gradient flows into it from both directions.

Everything is written out by hand. Attention is computed per batch element and
per head with a causal mask applied before the softmax, so a position can only
attend to itself and what came before it. The backward pass derives the softmax
Jacobian, the layer norm gradient with both the mean and variance terms, the
GELU derivative through its tanh approximation, and routes gradient correctly
through both sides of every residual.

## Verification

The backward pass is checked against numerical differentiation on **all sixteen
parameter tensors** — embeddings, both layer norms in a block, the QKV
projection and its bias, the attention output projection, both MLP layers and
the final norm. At each tensor's largest-magnitude entry, analytic and numerical
gradients agree to within **0.075%** in the worst case and exactly for several.

Checking at randomly chosen entries instead reports three apparent mismatches.
Those entries have gradients around 1e-6, where a central difference taken from
two float32 losses near 1.0 has no significant digits left. That is a limit of
the check, not of the derivation, which is why the check picks the largest
gradient in each tensor.

Attention is separately verified to be a proper causal distribution: the upper
triangle is exactly zero and every row sums to 1 within 3e-8.

Trained on the built-in Shakespeare sample it goes from a loss of about 3.8 to
**0.21** — a perplexity of 1.2 — in around a minute at 4,000 to 7,000 characters
per second on one thread, and starts returning fragments of the source.

## Notes

One HTML file. No libraries, no GPU, no weights fetched from anywhere. Paste
your own text and press train.

Built by [bxzex](https://bxzex.com).
