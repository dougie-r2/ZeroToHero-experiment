# ZeroToHero-experiment

This repo is from [ZeroToHero project](https://www.youtube.com/@AndrejKarpathy/playlists) by Andrej Karpathy.

I do some experiments for my own curiosity and to take away lessons.


### Lecture1 micro-grad


### Lecture2 bigram linear model


### Lecture3 MLP
- The goal is same as lecture 2 which is infer the next character given the context of name.
- Accept the MLP model from the [Bengio 2003 paper](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- Unlike lecture 2, use embedding to encode (one-hot enc has sparsity and othogonality)
- lecture2 was bigram, this one is 3-gram.
- pytorch has memory efficient api called `view` for reshaping tensors (read [pytorch internal blog post](http://blog.ezyang.com/2019/05/pytorch-internals/))
- `F.cross_entropy` does same operation of softmax and mean of Negative log likelihood. However it is more efficient in many ways than do seq of exp and norm and NLL.
  - forward and backward is much better in memory terms.
  - Numerically well behaved since it subtract the maximum value of the logits. 
- Find good initial learning rate with exponentially spaced and then apply first big step of that range of candidates later apply small one for preventing loss divergence 
- There are several ways to improve the model
  - Wider hidden layer dimension ([related blog post](posthttps://lilianweng.github.io/posts/2022-09-08-ntk/))
  - Larger embedding dimension
  - Increase batch size

![charater embedded](./imgs/output.png)

*Image of 2 dim Embedded characters. Quite amazed seeing that vowels are in the cluster. Letter y is more alike vowles which is make sense since the data is people name*

Ma Question  
1. what is the minimum loss we can get in theoretical way? not from empirical.
   It seems related to Entropy. Can we even calculate it?


### Lecture4 batchnorm