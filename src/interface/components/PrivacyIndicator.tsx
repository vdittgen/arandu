import { useCallback } from "react";
import { Globe, Lock } from "lucide-react";
import { dedupInvoke } from "../utils/requestDedup";
import { useAsyncData } from "../hooks/useAsyncData";

interface AppSettings {
  readonly llm_provider: string;
  readonly llm_model: string;
  readonly llm_host: string;
}

interface Routing {
  readonly mode: "local" | "remote";
  readonly label: string;
}

function providerDisplayName(provider: string, host: string): string {
  if (provider === "ollama") return "Ollama";
  if (provider === "anthropic") return "Anthropic";

  let hostname = host;
  try {
    hostname = new URL(host).hostname;
  } catch {
    // Leave as-is
  }
  const lc = hostname.toLowerCase();
  if (lc.includes("localhost") || lc.includes("127.0.0.1")) return "Local server";
  return hostname || "Remote";
}

function deriveRouting(settings: AppSettings): Routing {
  // The provider decides where traffic actually goes; the "keep
  // sensitive topics local" toggle does not change that. Checking the
  // toggle first used to show a green "Local" badge for a remote
  // provider whenever the toggle was on — reassuring the user while
  // traffic egressed.
  if (settings.llm_provider === "ollama") {
    return { mode: "local", label: "Local" };
  }

  const provider = providerDisplayName(settings.llm_provider, settings.llm_host);
  return { mode: "remote", label: provider };
}

function PrivacyIndicator() {
  const fetcher = useCallback(
    () => dedupInvoke<AppSettings>("get_settings"),
    [],
  );
  const { data, isLoading, error } = useAsyncData<AppSettings>(fetcher);

  if (isLoading && !data) {
    return (
      <div className="flex items-center gap-2">
        <span className="inline-flex items-center gap-2 rounded-pill bg-bg-2 px-3 py-1.5 text-[12.5px] font-medium text-muted">
          <span className="h-2 w-2 rounded-full bg-muted/40" />
          Checking…
        </span>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex items-center gap-2">
        <span className="inline-flex items-center gap-2 rounded-pill bg-amber-soft px-3 py-1.5 text-[12.5px] font-medium text-[oklch(0.36_0.10_70)]">
          <span className="flex h-4 w-4 items-center justify-center rounded-full bg-amber text-white">
            <Globe className="h-[9px] w-[9px]" strokeWidth={2.5} />
          </span>
          Unknown
        </span>
      </div>
    );
  }

  const routing = deriveRouting(data);
  const isLocal = routing.mode === "local";

  return (
    <div className="flex items-center gap-2">
      {isLocal ? (
        <span className="inline-flex items-center gap-2 rounded-pill bg-success-soft py-[5px] pl-2 pr-3 text-[12.5px] font-medium text-[oklch(0.36_0.10_155)]">
          <span className="flex h-4 w-4 items-center justify-center rounded-full bg-success text-white">
            <Lock className="h-[9px] w-[9px]" strokeWidth={2.5} />
          </span>
          Local
        </span>
      ) : (
        <span className="inline-flex items-center gap-2 rounded-pill bg-amber-soft py-[5px] pl-2 pr-3 text-[12.5px] font-medium text-[oklch(0.36_0.10_70)]">
          <span className="flex h-4 w-4 items-center justify-center rounded-full bg-amber text-white">
            <Globe className="h-[9px] w-[9px]" strokeWidth={2.5} />
          </span>
          {routing.label}
        </span>
      )}
    </div>
  );
}

export default PrivacyIndicator;
