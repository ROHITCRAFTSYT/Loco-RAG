# Security Policy

## Supported versions

Fixes land on `main`. There are no maintained release branches.

## Reporting a vulnerability

Report privately through GitHub's
[security advisory form](https://github.com/ROHITCRAFTSYT/Loco-RAG/security/advisories/new)
rather than opening a public issue. Expect an acknowledgement within seven days.

Include the affected endpoint or file, steps to reproduce, and what an
attacker gains.

## Scope

In scope:

- Path traversal or unrestricted write via the upload filename in
  `backend/app/routers/documents.py` — `file.filename` is attacker-controlled.
- Parser vulnerabilities reachable from an uploaded document
  (`backend/app/services/ingest.py`).
- Server-side request forgery through web search. `_extract()` in
  `backend/app/services/websearch.py` fetches result URLs with redirects
  followed and no host allowlist; see the note below.
- Leaking provider API keys through an API response, log line, or error body.
- Retrieved document content that reaches the frontend and executes there
  (stored XSS through `Markdown.tsx`).
- Vulnerable dependency versions pinned by this repository.

Out of scope:

- Vulnerabilities in Ollama, Chroma, LanceDB, or other upstream components —
  report those to their maintainers.
- Model output quality, hallucination, or refusal behaviour.
- Prompt injection that only changes what the model says back to the same
  user who supplied the text.

## Known limitation: outbound fetches are unrestricted

The web-search path retrieves URLs returned by the configured search backend
and follows redirects. There is no allowlist and no block on private address
ranges, so a search result — or a self-hosted SearXNG instance under someone
else's control — can make the backend request internal addresses such as
`169.254.169.254` or `localhost`.

This is a known limitation, not an accepted one. Until it is fixed, run the
backend where it has no reachable internal network, or leave web search off.

## Deployment expectations

The API has no authentication. `cors_origins` defaults to the local Vite dev
server, and the whole stack is meant to run on one machine against a local
model. Exposing it publicly means anyone who can reach the port can read every
collection and upload into it — put an authenticating proxy in front of it
first. Reports amounting to "my public deployment has no auth" are
configuration.

## Secrets

Keys are read from the environment; `.env` is git-ignored and only
`.env.example` is tracked. If a real key is ever committed, rotate it at the
provider — removing the commit is not enough.
