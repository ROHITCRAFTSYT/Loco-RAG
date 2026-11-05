# Agent mode

In agent mode the model is given two tools, a document search tool and a web
search tool, and decides for itself whether and when to call them. Tool rounds
run without streaming, and the loop stops after at most four rounds so a model
that keeps calling tools cannot run forever.

Sources returned by every tool call are collected and de-duplicated by id before
the final answer is streamed. If the model or endpoint does not support tool
calling, the agent falls back to answering directly without tools.
