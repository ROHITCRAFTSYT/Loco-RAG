import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { MessageSquarePlus, Pin, Search, Trash2 } from "lucide-react";
import { api } from "../lib/api";
import type { Conversation } from "../types";

interface Props {
  activeId: string | null;
  onSelect: (id: string) => void;
}

export function Sidebar({ activeId, onSelect }: Props) {
  const qc = useQueryClient();
  const [filter, setFilter] = useState("");
  const { data: conversations = [] } = useQuery({
    queryKey: ["conversations"],
    queryFn: api.listConversations,
  });

  const create = useMutation({
    mutationFn: () => api.createConversation(),
    onSuccess: (c) => {
      qc.invalidateQueries({ queryKey: ["conversations"] });
      onSelect(c.id);
    },
  });

  const del = useMutation({
    mutationFn: (id: string) => api.deleteConversation(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["conversations"] }),
  });

  const togglePin = useMutation({
    mutationFn: (c: Conversation) => api.updateConversation(c.id, { pinned: !c.pinned }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["conversations"] }),
  });

  const filtered = conversations.filter((c) =>
    c.title.toLowerCase().includes(filter.toLowerCase()),
  );

  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-border bg-surface">
      <div className="p-3">
        <button
          onClick={() => create.mutate()}
          className="flex w-full items-center justify-center gap-2 rounded-lg bg-accent
                     px-3 py-2 text-sm font-medium text-white hover:opacity-90"
        >
          <MessageSquarePlus size={16} /> New chat
        </button>
      </div>
      <div className="px-3 pb-2">
        <div className="flex items-center gap-2 rounded-lg border border-border bg-bg px-2">
          <Search size={14} className="text-muted" />
          <input
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Search chats"
            className="w-full bg-transparent py-1.5 text-sm outline-none"
          />
        </div>
      </div>
      <nav className="flex-1 space-y-0.5 overflow-y-auto px-2 pb-2">
        {filtered.map((c) => (
          <div
            key={c.id}
            onClick={() => onSelect(c.id)}
            className={`group flex cursor-pointer items-center gap-2 rounded-lg px-2.5 py-2 text-sm ${
              c.id === activeId ? "bg-accent/15 text-fg" : "text-muted hover:bg-bg hover:text-fg"
            }`}
          >
            <span className="flex-1 truncate">{c.title}</span>
            <button
              onClick={(e) => {
                e.stopPropagation();
                togglePin.mutate(c);
              }}
              className={`opacity-0 group-hover:opacity-100 ${c.pinned ? "text-accent opacity-100" : ""}`}
            >
              <Pin size={13} />
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                del.mutate(c.id);
              }}
              className="text-muted opacity-0 hover:text-red-400 group-hover:opacity-100"
            >
              <Trash2 size={13} />
            </button>
          </div>
        ))}
      </nav>
    </aside>
  );
}
