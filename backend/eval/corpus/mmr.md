# Diversifying results with MMR

Retrieval can return several chunks that say nearly the same thing, which wastes
the limited context. Maximal Marginal Relevance picks the final chunks greedily:
each pick balances similarity to the query against similarity to the chunks
already picked, so near-duplicate passages are skipped.

The balance is controlled by a lambda between zero and one. A lambda of one is
pure relevance order and a lambda of zero maximises diversity; the default of
one half weighs both. MMR is off by default and is enabled with an environment
flag.
