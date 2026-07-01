import { useEffect, useRef, useState } from "react";
import {
  Database,
  Globe,
  Loader2,
  Mic,
  Paperclip,
  Send,
  Sparkles,
  Square,
  X,
} from "lucide-react";
import { api } from "../lib/api";
import type { SendOptions } from "../types";

interface Props {
  conversationId: string | null;
  onSend: (text: string, opts: SendOptions) => void;
  onStop: () => void;
  streaming: boolean;
  ragEnabled: boolean;
  webEnabled: boolean;
  onToggleRag: () => void;
  onToggleWeb: () => void;
  draft?: string;
}

interface Attachment {
  filename: string;
  collection: string;
  chunks: number;
}

export function Composer({
  conversationId,
  onSend,
  onStop,
  streaming,
  ragEnabled,
  webEnabled,
  onToggleRag,
  onToggleWeb,
  draft,
}: Props) {
  const [text, setText] = useState("");
  const [agent, setAgent] = useState(false);
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [attaching, setAttaching] = useState(false);
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [micError, setMicError] = useState<string | null>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const recRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  useEffect(() => {
    if (draft !== undefined) setText(draft);
  }, [draft]);

  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 220) + "px";
  }, [text]);

  const submit = () => {
    const t = text.trim();
    if (!t || streaming) return;
    onSend(t, {
      attachment_collections: attachments.map((a) => a.collection),
      agent_mode: agent,
    });
    setText("");
  };

  const onPickFile = async (file: File) => {
    if (!conversationId) return;
    setAttaching(true);
    try {
      const res = await api.attachDocument(file, conversationId);
      setAttachments((a) => [
        ...a,
        { filename: res.filename, collection: res.collection, chunks: res.chunk_count },
      ]);
    } finally {
      setAttaching(false);
    }
  };

  const toggleRecording = async () => {
    if (recording) {
      recRef.current?.stop();
      return;
    }
    try {
      setMicError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      chunksRef.current = [];
      rec.ondataavailable = (e) => e.data.size && chunksRef.current.push(e.data);
      rec.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        setRecording(false);
        setTranscribing(true);
        try {
          const blob = new Blob(chunksRef.current, { type: "audio/webm" });
          const { text: spoken } = await api.transcribe(blob);
          if (spoken) setText((t) => (t ? `${t} ${spoken}` : spoken));
        } finally {
          setTranscribing(false);
        }
      };
      recRef.current = rec;
      rec.start();
      setRecording(true);
    } catch (err) {
      setRecording(false);
      setMicError(
        err instanceof DOMException && err.name === "NotAllowedError"
          ? "Microphone access was denied."
          : "Couldn't access the microphone.",
      );
    }
  };

  return (
    <div className="border-t border-border bg-bg/80 px-4 py-3 backdrop-blur">
      <div className="mx-auto max-w-3xl">
        <div className="mb-2 flex flex-wrap gap-2">
          <Toggle active={agent} onClick={() => setAgent((a) => !a)} icon={<Sparkles size={13} />}>
            Agent
          </Toggle>
          <Toggle active={ragEnabled} onClick={onToggleRag} icon={<Database size={13} />}>
            Documents (RAG)
          </Toggle>
          <Toggle active={webEnabled} onClick={onToggleWeb} icon={<Globe size={13} />}>
            Web search
          </Toggle>
        </div>

        {micError && (
          <div className="mb-2 text-xs text-red-400" role="alert">
            {micError}
          </div>
        )}

        {attachments.length > 0 && (
          <div className="mb-2 flex flex-wrap gap-2">
            {attachments.map((a, i) => (
              <span
                key={a.collection + i}
                className="flex items-center gap-1.5 rounded-full border border-accent bg-accent/10
                           px-2.5 py-1 text-xs text-accent"
              >
                <Paperclip size={11} /> {a.filename} · {a.chunks} chunks
                <button onClick={() => setAttachments((xs) => xs.filter((_, j) => j !== i))}>
                  <X size={12} />
                </button>
              </span>
            ))}
          </div>
        )}

        <div className="flex items-end gap-2 rounded-xl border border-border bg-surface p-2">
          <input
            ref={fileRef}
            type="file"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && onPickFile(e.target.files[0])}
          />
          <button
            onClick={() => fileRef.current?.click()}
            disabled={attaching || !conversationId}
            className="rounded-lg p-2 text-muted hover:text-fg disabled:opacity-40"
            title="Attach a file to this message"
          >
            {attaching ? <Loader2 size={16} className="animate-spin" /> : <Paperclip size={16} />}
          </button>
          <button
            onClick={toggleRecording}
            disabled={transcribing}
            className={`rounded-lg p-2 disabled:opacity-40 ${
              recording ? "animate-pulse text-red-400" : "text-muted hover:text-fg"
            }`}
            title="Dictate (local speech-to-text)"
          >
            {transcribing ? <Loader2 size={16} className="animate-spin" /> : <Mic size={16} />}
          </button>
          <textarea
            ref={taRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            rows={1}
            placeholder={
              recording ? "Listening…" : "Message your local model…  (Enter to send)"
            }
            className="max-h-56 flex-1 resize-none bg-transparent px-2 py-1.5 text-sm outline-none"
          />
          {streaming ? (
            <button
              onClick={onStop}
              className="rounded-lg border border-border bg-surface p-2 text-fg hover:bg-border"
              aria-label="Stop"
            >
              <Square size={16} />
            </button>
          ) : (
            <button
              onClick={submit}
              className="rounded-lg bg-accent p-2 text-white disabled:opacity-40"
              disabled={!text.trim()}
              aria-label="Send"
            >
              <Send size={16} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function Toggle({
  active,
  onClick,
  icon,
  children,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs transition-colors ${
        active
          ? "border-accent bg-accent/15 text-accent"
          : "border-border text-muted hover:text-fg"
      }`}
    >
      {icon}
      {children}
    </button>
  );
}
