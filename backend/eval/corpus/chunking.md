# Token-aware chunking

Documents are split into overlapping windows before they are embedded. The
chunker counts tokens with the `cl100k_base` encoding rather than characters or
words, so a window holds a predictable amount of model context regardless of
language or punctuation.

The default window is 512 tokens with 64 tokens of overlap between consecutive
chunks. Overlap keeps a sentence that straddles a window boundary retrievable
from both chunks, which reduces answers that get cut off mid-thought. Page
numbers from the source document are carried onto every chunk so that citations
can point a reader back to the exact page.
