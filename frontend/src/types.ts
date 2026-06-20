export interface Conversation {
  id: string;
  title: string;
  system_prompt: string | null;
  model: string | null;
  provider: string | null;
  temperature: number;
  top_p: number;
  max_tokens: number | null;
  rag_enabled: boolean;
  collection: string | null;
  web_search: boolean;
  pinned: boolean;
  created_at: string;
  updated_at: string;
}

export interface Source {
  id: string;
  document: string;
  collection?: string | null;
  page?: number | null;
  chunk_index?: number | null;
  score: number;
  text: string;
  kind: "document" | "web" | "memory";
  url?: string | null;
}

export interface Usage {
  prompt_tokens: number | null;
  completion_tokens: number | null;
  tps: number;
}

export interface Message {
  id: string;
  role: "system" | "user" | "assistant";
  content: string;
  sources?: Source[] | null;
  usage?: Usage | null;
  created_at?: string;
  // client-only flag for the in-flight streaming message
  streaming?: boolean;
}

export interface SendOptions {
  attachment_collections?: string[];
  agent_mode?: boolean;
}

export interface ModelInfo {
  id: string;
  provider: string;
  supports_vision: boolean;
}

export interface DocumentMeta {
  id: string;
  collection: string;
  filename: string;
  status: "pending" | "processing" | "ready" | "error";
  error: string | null;
  chunk_count: number;
  size_bytes: number;
  created_at: string;
}
