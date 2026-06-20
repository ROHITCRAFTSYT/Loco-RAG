import { useCallback, useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api, streamChat } from "../lib/api";
import type { Message, SendOptions, Source, Usage } from "../types";

let tmpId = 0;
const nextId = () => `tmp-${tmpId++}`;

export function useChat(conversationId: string | null) {
  const qc = useQueryClient();
  const [messages, setMessages] = useState<Message[]>([]);
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!conversationId) {
      setMessages([]);
      return;
    }
    api.messages(conversationId).then(setMessages);
  }, [conversationId]);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    setStreaming(false);
    setMessages((ms) => ms.map((m) => ({ ...m, streaming: false })));
  }, []);

  const send = useCallback(
    (content: string, opts: SendOptions = {}) => {
      if (!conversationId) return;
      const userMsg: Message = { id: nextId(), role: "user", content };
      const assistantMsg: Message = {
        id: nextId(),
        role: "assistant",
        content: "",
        streaming: true,
      };
      setMessages((ms) => [...ms, userMsg, assistantMsg]);
      setStreaming(true);

      const ctrl = new AbortController();
      abortRef.current = ctrl;
      let sources: Source[] | undefined;

      streamChat(
        conversationId,
        { content, ...opts },
        {
          onSources: (s) => {
            sources = s;
            setMessages((ms) =>
              ms.map((m) => (m.id === assistantMsg.id ? { ...m, sources: s } : m)),
            );
          },
          onToken: (t) =>
            setMessages((ms) =>
              ms.map((m) =>
                m.id === assistantMsg.id ? { ...m, content: m.content + t } : m,
              ),
            ),
          onUsage: (u: Usage) =>
            setMessages((ms) =>
              ms.map((m) => (m.id === assistantMsg.id ? { ...m, usage: u } : m)),
            ),
          onError: (e) =>
            setMessages((ms) =>
              ms.map((m) =>
                m.id === assistantMsg.id
                  ? { ...m, content: m.content + `\n\n_Error: ${e}_`, streaming: false }
                  : m,
              ),
            ),
          onDone: () => {
            setStreaming(false);
            setMessages((ms) =>
              ms.map((m) =>
                m.id === assistantMsg.id ? { ...m, streaming: false, sources } : m,
              ),
            );
            // Refresh the sidebar (auto-title on first message).
            qc.invalidateQueries({ queryKey: ["conversations"] });
          },
        },
        ctrl.signal,
      ).catch(() => setStreaming(false));
    },
    [conversationId, qc],
  );

  const editAndRerun = useCallback(
    async (message: Message, newContent: string, opts: SendOptions = {}) => {
      if (!conversationId) return;
      // Drop the edited message and everything after, then resend.
      if (!message.id.startsWith("tmp-")) {
        await api.deleteAfter(conversationId, message.id);
      }
      setMessages((ms) => {
        const idx = ms.findIndex((m) => m.id === message.id);
        return idx >= 0 ? ms.slice(0, idx) : ms;
      });
      send(newContent, opts);
    },
    [conversationId, send],
  );

  return { messages, streaming, send, stop, editAndRerun };
}
