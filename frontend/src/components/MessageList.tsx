import { useEffect, useRef, useState } from "react";
import { Bot, Pencil, User, Volume2, VolumeX } from "lucide-react";
import type { Message } from "../types";
import { Markdown } from "./Markdown";
import { SourcePanel } from "./SourcePanel";

function SpeakButton({ text }: { text: string }) {
  const [speaking, setSpeaking] = useState(false);
  if (!("speechSynthesis" in window) || !text) return null;
  const toggle = () => {
    if (speaking) {
      window.speechSynthesis.cancel();
      setSpeaking(false);
      return;
    }
    const u = new SpeechSynthesisUtterance(text);
    u.onend = () => setSpeaking(false);
    setSpeaking(true);
    window.speechSynthesis.speak(u);
  };
  return (
    <button
      onClick={toggle}
      className="flex items-center gap-1 text-[0.7rem] text-muted opacity-0
                 group-hover:opacity-100 hover:text-fg"
      title="Read aloud"
    >
      {speaking ? <VolumeX size={11} /> : <Volume2 size={11} />}
      {speaking ? "stop" : "speak"}
    </button>
  );
}

interface Props {
  messages: Message[];
  onEdit: (m: Message) => void;
}

export function MessageList({ messages, onEdit }: Props) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 px-4 py-6">
      {messages.map((m) => (
        <div key={m.id} className="group flex gap-3">
          <div
            className={`mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-md ${
              m.role === "user" ? "bg-accent/20 text-accent" : "bg-surface text-fg"
            }`}
          >
            {m.role === "user" ? <User size={16} /> : <Bot size={16} />}
          </div>
          <div className="min-w-0 flex-1">
            {m.role === "assistant" ? (
              <Markdown content={m.content || (m.streaming ? "▍" : "")} />
            ) : (
              <div className="whitespace-pre-wrap text-[0.95rem]">{m.content}</div>
            )}
            {m.sources && <SourcePanel sources={m.sources} />}
            <div className="mt-1 flex items-center gap-3">
              {m.usage && (
                <span className="text-[0.7rem] text-muted">
                  {m.usage.completion_tokens ?? "?"} tok · {m.usage.tps} tok/s
                </span>
              )}
              {m.role === "assistant" && !m.streaming && <SpeakButton text={m.content} />}
            </div>
            {m.role === "user" && (
              <button
                onClick={() => onEdit(m)}
                className="mt-1 flex items-center gap-1 text-[0.7rem] text-muted opacity-0
                           group-hover:opacity-100 hover:text-fg"
              >
                <Pencil size={11} /> edit & rerun
              </button>
            )}
          </div>
        </div>
      ))}
      <div ref={endRef} />
    </div>
  );
}
