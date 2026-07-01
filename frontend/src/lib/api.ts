import { fetchEventSource } from "@microsoft/fetch-event-source";
import type {
  Conversation,
  DocumentMeta,
  Message,
  ModelInfo,
  Source,
  Usage,
} from "../types";

const BASE = "";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json() as Promise<T>;
}

export const api = {
  health: () => fetch(`${BASE}/health`).then((r) => json<Record<string, string>>(r)),

  models: () => fetch(`${BASE}/api/models`).then((r) => json<ModelInfo[]>(r)),

  listConversations: () =>
    fetch(`${BASE}/api/conversations`).then((r) => json<Conversation[]>(r)),

  createConversation: (body: Partial<Conversation> = {}) =>
    fetch(`${BASE}/api/conversations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => json<Conversation>(r)),

  updateConversation: (id: string, body: Partial<Conversation>) =>
    fetch(`${BASE}/api/conversations/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => json<Conversation>(r)),

  deleteConversation: (id: string) =>
    fetch(`${BASE}/api/conversations/${id}`, { method: "DELETE" }).then((r) => json(r)),

  messages: (id: string) =>
    fetch(`${BASE}/api/conversations/${id}/messages`).then((r) => json<Message[]>(r)),

  deleteAfter: (convId: string, messageId: string) =>
    fetch(`${BASE}/api/conversations/${convId}/messages/after/${messageId}`, {
      method: "DELETE",
    }).then((r) => json(r)),

  // Documents
  listDocuments: (collection?: string) =>
    fetch(
      `${BASE}/api/documents${collection ? `?collection=${encodeURIComponent(collection)}` : ""}`,
    ).then((r) => json<DocumentMeta[]>(r)),

  listCollections: () =>
    fetch(`${BASE}/api/documents/collections`).then((r) => json<string[]>(r)),

  uploadDocument: (file: File, collection: string) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("collection", collection);
    return fetch(`${BASE}/api/documents/upload`, { method: "POST", body: fd }).then((r) =>
      json<{ document_id: string; status: string; chunk_count: number }>(r),
    );
  },

  deleteDocument: (id: string) =>
    fetch(`${BASE}/api/documents/${id}`, { method: "DELETE" }).then((r) => json(r)),

  // Ephemeral "talk to this doc" — ingests inline, returns its collection.
  attachDocument: (file: File, conversationId: string) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("conversation_id", conversationId);
    return fetch(`${BASE}/api/documents/attach`, { method: "POST", body: fd }).then((r) =>
      json<{ document_id: string; collection: string; filename: string; chunk_count: number }>(r),
    );
  },

  // Local speech-to-text.
  transcribe: (audio: Blob) => {
    const fd = new FormData();
    fd.append("file", audio, "speech.webm");
    return fetch(`${BASE}/api/voice/transcribe`, { method: "POST", body: fd }).then((r) =>
      json<{ text: string; language: string }>(r),
    );
  },
};

export interface StreamHandlers {
  onSources?: (s: Source[]) => void;
  onToken: (t: string) => void;
  onUsage?: (u: Usage) => void;
  onError?: (e: string) => void;
  onDone: () => void;
}

export function streamChat(
  conversationId: string,
  body: Record<string, unknown>,
  handlers: StreamHandlers,
  signal: AbortSignal,
) {
  return fetchEventSource(`${BASE}/api/conversations/${conversationId}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
    openWhenHidden: true,
    onmessage(ev) {
      switch (ev.event) {
        case "sources":
          handlers.onSources?.(JSON.parse(ev.data));
          break;
        case "token":
          handlers.onToken(ev.data);
          break;
        case "usage":
          handlers.onUsage?.(JSON.parse(ev.data));
          break;
        case "error":
          handlers.onError?.(ev.data);
          break;
        case "done":
          handlers.onDone();
          break;
      }
    },
    onerror(err) {
      handlers.onError?.(String(err));
      throw err; // stop retrying
    },
  });
}
