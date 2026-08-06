# Hybrid retrieval

Retrieval combines two signals. Dense retrieval embeds the query and finds the
nearest chunk vectors by cosine similarity, which captures meaning even when the
wording differs. Sparse retrieval runs BM25 keyword matching, which is strong on
exact terms, names, and rare tokens that an embedding can blur.

The two ranked lists are merged with Reciprocal Rank Fusion. RRF scores a chunk
by the sum of `1 / (k + rank)` across the lists it appears in, so a chunk that
ranks highly in both dense and sparse results rises to the top without either
signal's raw scores needing to be comparable. An optional cross-encoder reranker
can then reorder the fused candidates for a final precision boost.
