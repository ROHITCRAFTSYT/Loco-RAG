# Screenshots & media

Drop image/GIF files here using the **exact filenames** below and they will appear in the
main [README](../README.md) automatically — no further edits needed.

| Filename | What to capture | Suggested size |
|----------|-----------------|----------------|
| `chat-rag.png` | A streaming chat answer with the **source/citation panel expanded** | ~1280×800 |
| `knowledge-base.png` | The right-hand **Knowledge base** panel with a few ingested docs | ~1280×800 |
| `agent-mode.png` | A reply produced with **Agent** mode on (tools used) | ~1280×800 |
| `voice.png` | The composer mid-dictation, or an assistant message with **speak** | ~1280×800 |
| `demo.gif` | 8–12s screen recording: type a question → tokens stream → citations appear | ≤ 8 MB, ~1000px wide |

## Tips

- **Dark mode** generally screenshots better for a dev tool — toggle the sun/moon in the top bar.
- Trim browser chrome (or use the browser's device toolbar) so the UI fills the frame.
- For the GIF, keep it short and loop a single complete interaction. Tools: ScreenToGif
  (Windows), Kap (macOS), or `ffmpeg`.
- Optimize PNGs (e.g. https://tinypng.com) and GIFs to keep the repo lightweight.

Seed the sample doc first so RAG has something to cite:

```bash
docker compose exec backend python -m scripts.seed
# then ask: "How tall is the Eiffel Tower?" with Documents (RAG) enabled
```
