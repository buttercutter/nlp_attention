# nlp_attention
Some fun experiments around different types of NLP attention mechanisms

Done:
- [Group Query Attention](https://arxiv.org/abs/2305.13245)
- [Sliding Window Attention](https://arxiv.org/abs/2004.05150)
- [StreamingLLM](https://arxiv.org/abs/2309.17453)
- [Focused Transformer](https://arxiv.org/abs/2307.03170)
  
Todo:
- [Shaped Transformer](http://arxiv.org/abs/2306.17759) is still a WIP, `CovarianceTracker` class still needs some more work, and also need to derive Theorem 4.2 on page 9 and 30 of the paper.
- [Shift Short Attention](http://arxiv.org/abs/2309.12307) seems similar to [Blockwise Causal Attention, BCA](https://zhuanlan.zhihu.com/p/660073229)
- [Intention](https://arxiv.org/abs/2305.10203) requires some expensive matrix inversion operations on top of attention compute logic, this may be why it is not widely adopted until now.

Credit: [Rishikesh Magar](https://github.com/RishikeshMagar), [@ZickZack](https://twitter.com/Alex_Mattick)
