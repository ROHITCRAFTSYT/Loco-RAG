import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useDropzone } from "react-dropzone";
import { FileText, Loader2, Trash2, UploadCloud, X } from "lucide-react";
import { api } from "../lib/api";

interface Props {
  collection: string;
  onClose: () => void;
}

export function DocPanel({ collection, onClose }: Props) {
  const qc = useQueryClient();
  const [active, setActive] = useState(collection || "default");

  const { data: collections = ["default"] } = useQuery({
    queryKey: ["collections"],
    queryFn: api.listCollections,
  });

  const { data: docs = [] } = useQuery({
    queryKey: ["documents", active],
    queryFn: () => api.listDocuments(active),
    refetchInterval: (q) =>
      (q.state.data ?? []).some((d) => d.status === "processing" || d.status === "pending")
        ? 1500
        : false,
  });

  const upload = useMutation({
    mutationFn: (files: File[]) =>
      Promise.all(files.map((f) => api.uploadDocument(f, active))),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["documents", active] });
      qc.invalidateQueries({ queryKey: ["collections"] });
    },
  });

  const del = useMutation({
    mutationFn: (id: string) => api.deleteDocument(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["documents", active] }),
  });

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: (files) => upload.mutate(files),
  });

  return (
    <aside className="flex w-80 shrink-0 flex-col border-l border-border bg-surface">
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <h2 className="text-sm font-semibold">Knowledge base</h2>
        <button onClick={onClose} className="text-muted hover:text-fg">
          <X size={16} />
        </button>
      </div>

      <div className="px-4 py-3">
        <label className="mb-1 block text-xs text-muted">Collection</label>
        <input
          list="collections"
          value={active}
          onChange={(e) => setActive(e.target.value)}
          className="w-full rounded-lg border border-border bg-bg px-2 py-1.5 text-sm outline-none"
        />
        <datalist id="collections">
          {collections.map((c) => (
            <option key={c} value={c} />
          ))}
        </datalist>
      </div>

      <div className="px-4">
        <div
          {...getRootProps()}
          className={`flex cursor-pointer flex-col items-center gap-1 rounded-xl border-2
                      border-dashed p-5 text-center text-xs ${
                        isDragActive ? "border-accent bg-accent/10" : "border-border text-muted"
                      }`}
        >
          <input {...getInputProps()} />
          <UploadCloud size={20} />
          Drop files or click — PDF, DOCX, TXT, MD, CSV, code
        </div>
      </div>

      <div className="flex-1 space-y-1.5 overflow-y-auto p-4">
        {docs.map((d) => (
          <div
            key={d.id}
            className="flex items-center gap-2 rounded-lg border border-border bg-bg p-2 text-xs"
          >
            <FileText size={14} className="shrink-0 text-muted" />
            <div className="min-w-0 flex-1">
              <div className="truncate">{d.filename}</div>
              <div className="text-[0.7rem] text-muted">
                {d.status === "processing" || d.status === "pending" ? (
                  <span className="flex items-center gap-1">
                    <Loader2 size={10} className="animate-spin" /> {d.status}
                  </span>
                ) : d.status === "error" ? (
                  <span className="text-red-400">{d.error || "error"}</span>
                ) : (
                  `${d.chunk_count} chunks`
                )}
              </div>
            </div>
            <button onClick={() => del.mutate(d.id)} className="text-muted hover:text-red-400">
              <Trash2 size={13} />
            </button>
          </div>
        ))}
        {docs.length === 0 && (
          <p className="pt-6 text-center text-xs text-muted">No documents yet.</p>
        )}
      </div>
    </aside>
  );
}
