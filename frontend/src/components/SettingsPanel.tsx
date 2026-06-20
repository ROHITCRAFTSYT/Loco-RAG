import { useEffect, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { X } from "lucide-react";
import { api } from "../lib/api";
import type { Conversation } from "../types";

interface Props {
  conversation: Conversation;
  onClose: () => void;
}

export function SettingsPanel({ conversation, onClose }: Props) {
  const qc = useQueryClient();
  const [draft, setDraft] = useState(conversation);
  useEffect(() => setDraft(conversation), [conversation]);

  const save = useMutation({
    mutationFn: (body: Partial<Conversation>) => api.updateConversation(conversation.id, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["conversations"] });
      qc.invalidateQueries({ queryKey: ["conversation", conversation.id] });
    },
  });

  const update = (patch: Partial<Conversation>) => {
    setDraft((d) => ({ ...d, ...patch }));
    save.mutate(patch);
  };

  return (
    <div className="absolute inset-y-0 right-0 z-10 w-80 border-l border-border bg-surface shadow-xl">
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <h2 className="text-sm font-semibold">Conversation settings</h2>
        <button onClick={onClose} className="text-muted hover:text-fg">
          <X size={16} />
        </button>
      </div>
      <div className="space-y-4 p-4 text-sm">
        <Field label="System prompt">
          <textarea
            value={draft.system_prompt ?? ""}
            onChange={(e) => update({ system_prompt: e.target.value })}
            rows={4}
            className="w-full rounded-lg border border-border bg-bg px-2 py-1.5 text-sm outline-none"
            placeholder="You are a helpful assistant…"
          />
        </Field>
        <Field label={`Temperature: ${draft.temperature.toFixed(2)}`}>
          <input
            type="range"
            min={0}
            max={2}
            step={0.05}
            value={draft.temperature}
            onChange={(e) => update({ temperature: parseFloat(e.target.value) })}
            className="w-full"
          />
        </Field>
        <Field label="Max tokens (blank = model default)">
          <input
            type="number"
            value={draft.max_tokens ?? ""}
            onChange={(e) =>
              update({ max_tokens: e.target.value ? parseInt(e.target.value) : null })
            }
            className="w-full rounded-lg border border-border bg-bg px-2 py-1.5 text-sm outline-none"
          />
        </Field>
        <Field label="Default collection for RAG">
          <input
            value={draft.collection ?? ""}
            onChange={(e) => update({ collection: e.target.value || null })}
            placeholder="default"
            className="w-full rounded-lg border border-border bg-bg px-2 py-1.5 text-sm outline-none"
          />
        </Field>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={draft.rag_enabled}
            onChange={(e) => update({ rag_enabled: e.target.checked })}
          />
          Enable document RAG by default
        </label>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={draft.web_search}
            onChange={(e) => update({ web_search: e.target.checked })}
          />
          Enable web search by default
        </label>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="mb-1 block text-xs text-muted">{label}</label>
      {children}
    </div>
  );
}
