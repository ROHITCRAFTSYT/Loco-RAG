# Voice input and output

Speech-to-text runs locally with faster-whisper. The whisper model is loaded
lazily on the first transcription request so it never slows down startup, and
it runs on the CPU with int8 quantisation to keep memory use low. The model size
is configurable, from tiny up to large.

Text-to-speech needs no server component at all: replies are read aloud by the
browser's built-in speech synthesis, so no audio ever leaves the machine.
