# Uploading documents

Documents are added by drag-and-drop upload. PDF and DOCX files are parsed with
their page structure, while text, markdown, CSV and source code are read as a
single block of text. Uploads are capped at 50 megabytes by default, and a
larger file is rejected with HTTP 413 before it can exhaust server memory.

Ingestion runs in the background, so the upload request returns immediately and
the document moves from pending to processing to ready. Attaching a file to a
single chat message instead ingests it synchronously into a private
per-conversation collection, so the next message can search just that file.
