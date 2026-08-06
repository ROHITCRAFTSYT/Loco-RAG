# Web search tool

When a question needs current or external information, an optional web-search
tool fetches live results. The provider is either DuckDuckGo or a self-hosted
SearXNG instance, chosen by configuration, so the feature can run fully
self-hosted with no third-party account.

Each result page is fetched and its main text is extracted with trafilatura,
discarding navigation and boilerplate. The extracted passages become the same
kind of Source objects that document retrieval produces, so they flow through
one citation-aware context builder. That means the model cites a web page and an
uploaded document with the same bracketed `[n]` markers.
