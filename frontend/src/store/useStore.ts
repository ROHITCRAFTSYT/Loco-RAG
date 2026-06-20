import { create } from "zustand";
import { persist } from "zustand/middleware";

interface UIState {
  theme: "dark" | "light";
  selectedModel: string | null;
  selectedProvider: string | null;
  docPanelOpen: boolean;
  setTheme: (t: "dark" | "light") => void;
  toggleTheme: () => void;
  setModel: (model: string, provider: string) => void;
  setDocPanelOpen: (v: boolean) => void;
}

export const useStore = create<UIState>()(
  persist(
    (set) => ({
      theme: "dark",
      selectedModel: null,
      selectedProvider: null,
      docPanelOpen: false,
      setTheme: (theme) => {
        document.documentElement.classList.toggle("dark", theme === "dark");
        set({ theme });
      },
      toggleTheme: () =>
        set((s) => {
          const theme = s.theme === "dark" ? "light" : "dark";
          document.documentElement.classList.toggle("dark", theme === "dark");
          return { theme };
        }),
      setModel: (selectedModel, selectedProvider) => set({ selectedModel, selectedProvider }),
      setDocPanelOpen: (docPanelOpen) => set({ docPanelOpen }),
    }),
    { name: "llm-rag-ui" },
  ),
);
