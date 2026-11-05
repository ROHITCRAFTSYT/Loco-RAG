# Conversation memory

Long conversations would eventually overflow the model's context window, so
older turns are compressed. The six most recent messages are always sent
verbatim. Once the running history exceeds a token budget of 3000 tokens by
default, everything older than those recent messages is summarized by the model
into terse bullet points covering facts, decisions, and user preferences.

The summary is injected as a system note ahead of the recent turns. If the
summarization call fails, the app drops the older turns and sends only the
recent ones rather than risking a context overflow.
