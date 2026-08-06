# Embedding model

Each chunk is turned into a dense vector by a local embedding model. The default
is `BAAI/bge-small-en-v1.5` served through fastembed, which runs on CPU and
needs no network access or API key.

The same model embeds both stored chunks and incoming queries, so query and
document vectors live in one shared space where cosine similarity is meaningful.
Because the model is small it embeds quickly, which matters when a whole
collection has to be re-indexed after a backend switch. Swapping in a larger
embedding model trades ingest and query latency for retrieval accuracy.
