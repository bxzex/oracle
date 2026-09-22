# Oracle

A tiny character-level transformer that trains in your browser on whatever text you paste in, and shows you where its attention is going as it reads.

https://bxzex.github.io/oracle/

It's 2 blocks, 3 heads and a 48 character context, about 61k parameters by default. Attention, layer norm, GELU, the MLP and Adam are all written by hand, backward pass included.

I checked the gradients of every parameter tensor against numerical differences. The worst one was off by 0.075%. On the Shakespeare sample, loss drops from about 3.8 to 0.2 in roughly a minute, and it starts producing recognisable fragments.
