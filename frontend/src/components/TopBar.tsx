import { useQuery } from "@tanstack/react-query";
import { Database, Moon, Settings, Sun } from "lucide-react";
import { api } from "../lib/api";
import { useStore } from "../store/useStore";
import type { Conversation } from "../types";

interface Props {
  conversation: Conversation | null;
  onModelChange: (model: string, provider: string) => void;
  onToggleSettings: () => void;
}

export function TopBar({ conversation, onModelChange, onToggleSettings }: Props) {
  const { theme, toggleTheme, docPanelOpen, setDocPanelOpen } = useStore();
  const { data: models = [] } = useQuery({ queryKey: ["models"], queryFn: api.models });

  const current = conversation?.model ?? "";

  return (
    <header className="flex items-center gap-3 border-b border-border bg-surface px-4 py-2">
      <h1 className="text-sm font-semibold">{conversation?.title ?? "Local LLM Chat"}</h1>
      <div className="ml-auto flex items-center gap-2">
        <select
          value={current}
          onChange={(e) => {
            const m = models.find((x) => x.id === e.target.value);
            if (m) onModelChange(m.id, m.provider);
          }}
          className="rounded-lg border border-border bg-bg px-2 py-1.5 text-xs outline-none"
        >
          <option value="" disabled>
            {models.length ? "Select model" : "No models found"}
          </option>
          {models.map((m) => (
            <option key={`${m.provider}:${m.id}`} value={m.id}>
              {m.id} · {m.provider}
              {m.supports_vision ? " 👁" : ""}
            </option>
          ))}
        </select>
        <button
          onClick={() => setDocPanelOpen(!docPanelOpen)}
          className={`rounded-lg border border-border p-1.5 ${
            docPanelOpen ? "text-accent" : "text-muted hover:text-fg"
          }`}
          title="Knowledge base"
        >
          <Database size={16} />
        </button>
        <button
          onClick={onToggleSettings}
          className="rounded-lg border border-border p-1.5 text-muted hover:text-fg"
          title="Conversation settings"
        >
          <Settings size={16} />
        </button>
        <button
          onClick={toggleTheme}
          className="rounded-lg border border-border p-1.5 text-muted hover:text-fg"
          title="Toggle theme"
        >
          {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
        </button>
      </div>
    </header>
  );
}
