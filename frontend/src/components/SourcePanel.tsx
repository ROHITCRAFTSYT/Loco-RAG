import { useState } from "react";
import { ChevronDown, FileText, Globe } from "lucide-react";
import type { Source } from "../types";

export function SourcePanel({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false);
  if (!sources?.length) return null;
  return (
    <div className="mt-2 rounded-lg border border-border bg-surface/60">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-3 py-2 text-xs text-muted hover:text-fg"
      >
        <ChevronDown
          size={14}
          className={`transition-transform ${open ? "rotate-180" : ""}`}
        />
        {sources.length} source{sources.length > 1 ? "s" : ""}
      </button>
      {open && (
        <ol className="space-y-2 px-3 pb-3">
          {sources.map((s, i) => (
            <li key={s.id} className="rounded border border-border bg-bg p-2 text-xs">
              <div className="mb-1 flex items-center gap-1.5 font-medium">
                <span className="text-accent">[{i + 1}]</span>
                {s.kind === "web" ? <Globe size={12} /> : <FileText size={12} />}
                {s.url ? (
                  <a href={s.url} target="_blank" rel="noreferrer" className="truncate">
                    {s.document}
                  </a>
                ) : (
                  <span className="truncate">
                    {s.document}
                    {s.page ? ` · p.${s.page}` : ""}
                  </span>
                )}
                <span className="ml-auto text-muted">{s.score.toFixed(3)}</span>
              </div>
              <p className="line-clamp-3 text-muted">{s.text}</p>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
