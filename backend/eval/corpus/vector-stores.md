# Vector stores

Two interchangeable vector backends are supported, selected with the
`VECTOR_BACKEND` environment variable. ChromaDB is the default: it is embedded,
persistent, and requires zero setup. LanceDB is a fast, file-based alternative
that also runs with no server process.

Both backends implement the same `VectorStore` interface — upsert, query, and a
full-collection scan used to build the keyword index — so switching between them
is a configuration change rather than a code change. The only cost of switching
is re-ingesting the documents, because vectors written by one backend are not
readable by the other.
