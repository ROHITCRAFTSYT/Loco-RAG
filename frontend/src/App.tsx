import { useEffect, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Composer } from "./components/Composer";
import { DocPanel } from "./components/DocPanel";
import { MessageList } from "./components/MessageList";
import { SettingsPanel } from "./components/SettingsPanel";
import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import { useChat } from "./hooks/useChat";
import { api } from "./lib/api";
import { useStore } from "./store/useStore";
import type { Conversation, Message, SendOptions } from "./types";

export default function App() {
  const qc = useQueryClient();
  const { theme, setTheme, docPanelOpen, setDocPanelOpen } = useStore();
  const [activeId, setActiveId] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [draft, setDraft] = useState<string | undefined>(undefined);
  const [editing, setEditing] = useState<Message | null>(null);

  // Apply persisted theme on mount.
  useEffect(() => setTheme(theme), []); // eslint-disable-line

  const { data: conversations = [] } = useQuery({
    queryKey: ["conversations"],
    queryFn: api.listConversations,
  });

  // Auto-select or auto-create a conversation.
  useEffect(() => {
    if (activeId) return;
    if (conversations.length) setActiveId(conversations[0].id);
    else api.createConversation().then((c) => {
      qc.invalidateQueries({ queryKey: ["conversations"] });
      setActiveId(c.id);
    });
  }, [conversations, activeId, qc]);

  const conversation: Conversation | null = useMemo(
    () => conversations.find((c) => c.id === activeId) ?? null,
    [conversations, activeId],
  );

  const { messages, streaming, send, stop, editAndRerun } = useChat(activeId);

  const handleSend = (text: string, opts: SendOptions) => {
    if (editing) {
      editAndRerun(editing, text, opts);
      setEditing(null);
      setDraft(undefined);
      return;
    }
    send(text, opts);
  };

  const onModelChange = (model: string, provider: string) => {
    if (!conversation) return;
    api.updateConversation(conversation.id, { model, provider }).then(() =>
      qc.invalidateQueries({ queryKey: ["conversations"] }),
    );
  };

  const patchConv = (patch: Partial<Conversation>) => {
    if (!conversation) return;
    api.updateConversation(conversation.id, patch).then(() =>
      qc.invalidateQueries({ queryKey: ["conversations"] }),
    );
  };

  return (
    <div className="flex h-full">
      <Sidebar activeId={activeId} onSelect={setActiveId} />
      <main className="relative flex min-w-0 flex-1 flex-col">
        <TopBar
          conversation={conversation}
          onModelChange={onModelChange}
          onToggleSettings={() => setShowSettings((s) => !s)}
        />
        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <Empty />
          ) : (
            <MessageList
              messages={messages}
              onEdit={(m) => {
                setEditing(m);
                setDraft(m.content);
              }}
            />
          )}
        </div>
        <Composer
          conversationId={activeId}
          onSend={handleSend}
          onStop={stop}
          streaming={streaming}
          draft={draft}
          ragEnabled={!!conversation?.rag_enabled}
          webEnabled={!!conversation?.web_search}
          onToggleRag={() => patchConv({ rag_enabled: !conversation?.rag_enabled })}
          onToggleWeb={() => patchConv({ web_search: !conversation?.web_search })}
        />
        {showSettings && conversation && (
          <SettingsPanel conversation={conversation} onClose={() => setShowSettings(false)} />
        )}
      </main>
      {docPanelOpen && (
        <DocPanel
          collection={conversation?.collection ?? "default"}
          onClose={() => setDocPanelOpen(false)}
        />
      )}
    </div>
  );
}

function Empty() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 text-center text-muted">
      <p className="text-lg font-medium text-fg">Local LLM Chat with RAG</p>
      <p className="max-w-sm text-sm">
        Pick a model, optionally toggle Documents (RAG) or Web search, and start chatting.
        Everything runs locally.
      </p>
    </div>
  );
}
