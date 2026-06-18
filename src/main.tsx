import React, { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { check, Update } from "@tauri-apps/plugin-updater";
import { relaunch } from "@tauri-apps/plugin-process";
import {
  AlertCircle,
  CheckCircle2,
  Copy,
  Film,
  FolderOpen,
  Gauge,
  MessageSquare,
  Pencil,
  Play,
  RefreshCw,
  Send,
  Settings2,
  Trash2,
} from "lucide-react";
import logoBrutalist from "./assets/logo-brutalist.png";
import "./styles.css";

type FrameNote = {
  image_path: string;
  time: number;
  description: string;
};

type AnalyzeResponse = {
  analysis: string;
  analysis_path: string;
  study_markdown_path: string | null;
  frames_markdown_path: string | null;
  video_path: string;
  media_kind: "audio" | "video" | null;
  subtitle_path: string | null;
  thumbnail_path: string | null;
  subtitles: SubtitleCue[];
  frames: FrameNote[];
  stdout: string;
  stderr: string;
};

type SetupResponse = {
  stdout: string;
  stderr: string;
};

type SubtitleCue = {
  index: number;
  start: number;
  end: number;
  text: string;
};

type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

type ChatResponse = {
  message: ChatMessage;
};

type LibraryItem = {
  id: string;
  title: string;
  group: string;
  created_at: number;
  analysis: string;
  analysis_path: string;
  study_markdown_path: string | null;
  frames_markdown_path: string | null;
  video_path: string;
  media_kind: "audio" | "video" | null;
  subtitle_path: string | null;
  thumbnail_path: string | null;
  subtitles: SubtitleCue[];
  frames: FrameNote[];
};

type OllamaModel = {
  name: string;
};

const defaultPrompt =
  "Analyze this media. Write your response in the SAME language as the spoken content/transcript (English for an English video, Korean for a Korean video). Summarize the main topic, key claims, important visual or audio details, spoken content, and study points.";

function formatTime(seconds: number) {
  const whole = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(whole / 60);
  const rest = whole % 60;
  return `${minutes}:${rest.toString().padStart(2, "0")}`;
}

function App() {
  const mediaRef = useRef<HTMLMediaElement | null>(null);
  const [activeView, setActiveView] = useState<"study" | "chat">("study");

  const [url, setUrl] = useState("");
  const [model, setModel] = useState("gemma4:12b");
  const [batchProgress, setBatchProgress] = useState("");
  const [batchErrors, setBatchErrors] = useState<{ source: string; error: string }[]>([]);
  const [queueCount, setQueueCount] = useState(0);
  const pendingRef = useRef<string[]>([]);
  const processingRef = useRef(false);
  const [maxHeight, setMaxHeight] = useState(360);
  const [maxTokens, setMaxTokens] = useState(1024);
  const [frameCount, setFrameCount] = useState(32);
  const [outputLanguage, setOutputLanguage] = useState("auto");
  const [prompt, setPrompt] = useState(defaultPrompt);
  const [keepThinking, setKeepThinking] = useState(false);
  const [repeatCue, setRepeatCue] = useState<SubtitleCue | null>(null);
  const [repeatMode, setRepeatMode] = useState<"loop" | "once">("loop");
  const [playbackRate, setPlaybackRate] = useState(1);

  const [status, setStatus] = useState<"idle" | "running" | "done" | "error">(
    "idle",
  );
  const [setupStatus, setSetupStatus] = useState<
    "idle" | "running" | "done" | "error"
  >("idle");
  const [setupLog, setSetupLog] = useState("");
  const [error, setError] = useState("");
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [library, setLibrary] = useState<LibraryItem[]>([]);
  const [libraryError, setLibraryError] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const [editGroup, setEditGroup] = useState("");
  const [draggedId, setDraggedId] = useState<string | null>(null);
  const [dragOverGroup, setDragOverGroup] = useState<string | null>(null);

  const [update, setUpdate] = useState<Update | null>(null);
  const [updateStatus, setUpdateStatus] = useState<
    "idle" | "available" | "downloading" | "error"
  >("idle");
  const [updateError, setUpdateError] = useState("");

  const [chatEndpoint, setChatEndpoint] = useState(
    "http://127.0.0.1:11434/v1/chat/completions",
  );
  const [chatModel, setChatModel] = useState("gemma4:latest");
  const [ollamaModels, setOllamaModels] = useState<OllamaModel[]>([]);
  const [ollamaStatus, setOllamaStatus] = useState("");
  const [chatInput, setChatInput] = useState("");
  const [chatStatus, setChatStatus] = useState<"idle" | "running" | "error">(
    "idle",
  );
  const [chatError, setChatError] = useState("");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      role: "system",
      content:
        "You are an English AI study tutor that helps users learn from videos, audio, transcripts, and subtitles.",
    },
  ]);

  const sources = useMemo(
    () =>
      url
        .split("\n")
        .map((line) => line.trim())
        .filter((line) => line.length > 0),
    [url],
  );

  // You can enqueue any time — even while an analysis is already running.
  const canRun = useMemo(() => sources.length > 0, [sources]);

  useEffect(() => {
    void refreshLibrary();
    void refreshOllamaModels();

    let unlisten: (() => void) | undefined;
    getCurrentWebview()
      .onDragDropEvent((event) => {
        const kind = event.payload.type;
        if (kind === "enter" || kind === "over") {
          setIsDragging(true);
        } else if (kind === "leave") {
          setIsDragging(false);
        } else if (kind === "drop") {
          setIsDragging(false);
          const paths = event.payload.paths ?? [];
          if (paths.length > 0) {
            // Append every dropped file (video or audio) to the batch list,
            // one per line, keeping anything already typed.
            setUrl((prev) => {
              const existing = prev.trim();
              const added = paths.join("\n");
              return existing ? `${existing}\n${added}` : added;
            });
            setActiveView("study");
          }
        }
      })
      .then((handler) => {
        unlisten = handler;
      })
      .catch((err) => setLibraryError(String(err)));

    return () => {
      if (unlisten) unlisten();
    };
  }, []);

  async function refreshLibrary() {
    try {
      const items = await invoke<LibraryItem[]>("list_library");
      setLibrary(items);
      setLibraryError("");
    } catch (err) {
      setLibraryError(String(err));
    }
  }

  async function refreshOllamaModels() {
    try {
      const models = await invoke<OllamaModel[]>("list_ollama_models");
      setOllamaModels(models);
      setOllamaStatus(`${models.length} Ollama models found.`);
      if (!models.some((model) => model.name === chatModel)) {
        const gemma = models.find((model) => model.name.startsWith("gemma4"));
        if (gemma) {
          setChatModel(gemma.name);
        }
      }
    } catch (err) {
      setOllamaStatus(String(err));
    }
  }

  // Check GitHub Releases for a newer signed build on startup. In dev (or offline)
  // the updater isn't wired up, so failures are logged, not shown to the user.
  useEffect(() => {
    void (async () => {
      try {
        const found = await check();
        if (found) {
          setUpdate(found);
          setUpdateStatus("available");
        }
      } catch (err) {
        console.warn("Update check skipped:", err);
      }
    })();
  }, []);

  async function installUpdate() {
    if (!update) return;
    setUpdateStatus("downloading");
    setUpdateError("");
    try {
      await update.downloadAndInstall();
      await relaunch();
    } catch (err) {
      setUpdateError(String(err));
      setUpdateStatus("error");
    }
  }

  function runAnalysis(event: FormEvent) {
    event.preventDefault();
    if (sources.length === 0) return;
    // Append to the work queue and clear the box so more links can be added
    // (even while an earlier analysis is still running).
    pendingRef.current.push(...sources);
    setQueueCount(pendingRef.current.length);
    setUrl("");
    setActiveView("study");
    void processQueue();
  }

  async function processQueue() {
    if (processingRef.current) return; // a worker is already draining the queue
    processingRef.current = true;
    setStatus("running");
    setError("");

    while (pendingRef.current.length > 0) {
      const source = pendingRef.current.shift()!;
      setQueueCount(pendingRef.current.length);
      setBatchProgress(
        pendingRef.current.length > 0
          ? `${source} · ${pendingRef.current.length} queued`
          : source,
      );
      try {
        const response = await invoke<AnalyzeResponse>("analyze_youtube", {
          request: {
            url: source,
            model: model.trim(),
            max_height: maxHeight,
            max_new_tokens: maxTokens,
            prompt,
            keep_thinking: keepThinking,
            frame_count: frameCount,
            output_language: outputLanguage,
          },
        });
        setResult(response);
        await refreshLibrary();
      } catch (err) {
        setBatchErrors((prev) => [...prev, { source, error: String(err) }]);
      }
    }

    processingRef.current = false;
    setBatchProgress("");
    setStatus("done");
  }

  function cancelQueue() {
    // Drop everything still waiting. The item currently being analyzed finishes
    // on its own (its subprocess can't be interrupted mid-run), then the worker
    // stops because the queue is empty.
    const dropped = pendingRef.current.length;
    pendingRef.current = [];
    setQueueCount(0);
    if (dropped > 0) {
      setBatchProgress(processingRef.current ? "finishing current item…" : "");
    }
  }

  async function openFolder(path: string) {
    try {
      await invoke("open_path", { path });
    } catch (err) {
      setError(String(err));
    }
  }

  async function deleteLibraryItem(id: string, event: React.MouseEvent) {
    event.stopPropagation();
    try {
      const items = await invoke<LibraryItem[]>("delete_library_item", { id });
      setLibrary(items);
      setLibraryError("");
    } catch (err) {
      setLibraryError(String(err));
    }
  }

  async function moveToGroup(id: string, group: string) {
    const item = library.find((it) => it.id === id);
    if (!item || (item.group || "") === group) return;
    try {
      const items = await invoke<LibraryItem[]>("update_library_item", {
        id,
        title: item.title,
        group,
      });
      setLibrary(items);
      setLibraryError("");
    } catch (err) {
      setLibraryError(String(err));
    }
  }

  function startEdit(item: LibraryItem, event: React.MouseEvent) {
    event.stopPropagation();
    setEditingId(item.id);
    setEditTitle(item.title);
    setEditGroup(item.group || "");
  }

  async function saveEdit(event: React.MouseEvent) {
    event.stopPropagation();
    if (!editingId) return;
    try {
      const items = await invoke<LibraryItem[]>("update_library_item", {
        id: editingId,
        title: editTitle,
        group: editGroup,
      });
      setLibrary(items);
      setEditingId(null);
      setLibraryError("");
    } catch (err) {
      setLibraryError(String(err));
    }
  }

  // Build an ordered list of groups (named groups first, ungrouped last).
  const groupedLibrary = useMemo(() => {
    const groups = new Map<string, LibraryItem[]>();
    for (const item of library) {
      const key = item.group?.trim() || "";
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)!.push(item);
    }
    const named = [...groups.keys()].filter((k) => k !== "").sort();
    const ordered = groups.has("") ? [...named, ""] : named;
    return ordered.map((key) => ({ group: key, items: groups.get(key)! }));
  }, [library]);

  async function openLibraryItem(item: LibraryItem) {
    try {
      const response = await invoke<AnalyzeResponse>("load_library_item", {
        id: item.id,
      });
      setResult(response);
      setUrl(response.video_path);
      setStatus("done");
      setError("");
      setRepeatCue(null);
      setActiveView("study");
    } catch (err) {
      setError(String(err));
      setStatus("error");
    }
  }

  async function runSetup() {
    setSetupStatus("running");
    setSetupLog("Preparing the local runtime and models.");
    try {
      const response = await invoke<SetupResponse>("setup_environment", {
        request: {
          gemma_model: model.trim(),
        },
      });
      setSetupLog([response.stdout, response.stderr].filter(Boolean).join("\n"));
      setSetupStatus("done");
    } catch (err) {
      setSetupLog(String(err));
      setSetupStatus("error");
    }
  }

  async function sendChat(event: FormEvent) {
    event.preventDefault();
    const content = chatInput.trim();
    if (!content || chatStatus === "running") return;

    const context = result
      ? `\n\nCurrent media analysis:\n${result.analysis}\n\nSubtitle excerpt:\n${result.subtitles
          .slice(0, 20)
          .map((cue) => `${formatTime(cue.start)} ${cue.text}`)
          .join("\n")}`
      : "";

    const nextMessages: ChatMessage[] = [
      ...chatMessages,
      { role: "user", content: content + context },
    ];
    setChatMessages(nextMessages);
    setChatInput("");
    setChatStatus("running");
    setChatError("");

    try {
      const response = await invoke<ChatResponse>("chat_gemma", {
        request: {
          endpoint: chatEndpoint.trim(),
          api_key: "",
          model: chatModel.trim(),
          messages: nextMessages,
        },
      });
      setChatMessages([...nextMessages, response.message]);
      setChatStatus("idle");
    } catch (err) {
      setChatError(String(err));
      setChatStatus("error");
    }
  }

  function playCue(cue: SubtitleCue) {
    const media = mediaRef.current;
    if (!media) return;
    setRepeatCue(cue);
    media.playbackRate = playbackRate;
    media.currentTime = cue.start;
    void media.play();
  }

  function handleTimeUpdate() {
    const media = mediaRef.current;
    if (!media || !repeatCue) return;
    if (media.currentTime >= repeatCue.end) {
      if (repeatMode === "loop") {
        media.currentTime = repeatCue.start;
        void media.play();
      } else {
        media.pause();
        setRepeatCue(null);
      }
    }
  }

  const videoSrc = result?.video_path ? convertFileSrc(result.video_path) : "";
  const visibleChatMessages = chatMessages.filter((message) => message.role !== "system");

  return (
    <main className="app-shell">
      {update && (
        <div className="update-banner" data-status={updateStatus}>
          <span>
            🚀 Version <strong>{update.version}</strong> is available.
            {updateStatus === "downloading" && " Downloading…"}
            {updateStatus === "error" && ` Update failed: ${updateError}`}
          </span>
          <span className="update-banner-actions">
            <button
              type="button"
              onClick={installUpdate}
              disabled={updateStatus === "downloading"}
            >
              {updateStatus === "downloading" ? "Updating…" : "Update & Restart"}
            </button>
            <button
              type="button"
              className="update-later"
              onClick={() => setUpdate(null)}
              disabled={updateStatus === "downloading"}
            >
              Later
            </button>
          </span>
        </div>
      )}
      {isDragging && (
        <div className="drop-overlay">
          <strong>Drop video / audio files</strong>
          <span>They will be added to the batch list</span>
        </div>
      )}
      <section className="workspace">
        <aside className="library-rail">
          <div className="library-header">
            <strong>Study Library</strong>
            <button type="button" onClick={refreshLibrary}>
              Refresh
            </button>
          </div>
          {libraryError && <div className="library-error">{libraryError}</div>}
          {library.length > 0 ? (
            <div className="library-list">
              {groupedLibrary.map(({ group, items }) => (
                <div
                  className="library-group"
                  key={group || "__ungrouped__"}
                  data-dragover={dragOverGroup === (group || "")}
                  onDragOver={(e) => {
                    if (draggedId) {
                      e.preventDefault();
                      setDragOverGroup(group || "");
                    }
                  }}
                  onDragLeave={() => setDragOverGroup(null)}
                  onDrop={(e) => {
                    e.preventDefault();
                    if (draggedId) void moveToGroup(draggedId, group || "");
                    setDraggedId(null);
                    setDragOverGroup(null);
                  }}
                >
                  <div className="library-group-title">
                    {group || "Ungrouped"} · {items.length}
                  </div>
                  {items.map((item) =>
                    editingId === item.id ? (
                      <div className="library-edit" key={item.id}>
                        <input
                          value={editTitle}
                          onChange={(e) => setEditTitle(e.target.value)}
                          placeholder="Title"
                          spellCheck={false}
                        />
                        <input
                          value={editGroup}
                          onChange={(e) => setEditGroup(e.target.value)}
                          placeholder="Group (move to…)"
                          spellCheck={false}
                        />
                        <div className="library-edit-actions">
                          <button type="button" onClick={saveEdit}>
                            Save
                          </button>
                          <button type="button" onClick={() => setEditingId(null)}>
                            Cancel
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div
                        className="library-item"
                        key={item.id}
                        role="button"
                        tabIndex={0}
                        draggable
                        onDragStart={() => setDraggedId(item.id)}
                        onDragEnd={() => {
                          setDraggedId(null);
                          setDragOverGroup(null);
                        }}
                        onClick={() => openLibraryItem(item)}
                      >
                        {item.thumbnail_path ? (
                          <img src={convertFileSrc(item.thumbnail_path)} alt="" />
                        ) : (
                          <span className="library-thumb">{item.media_kind || "media"}</span>
                        )}
                        <span className="library-item-body">
                          <strong>{item.title}</strong>
                          <small>
                            {item.media_kind || "video"} · {item.subtitles.length} lines
                          </small>
                          {item.analysis && (
                            <span className="library-item-summary">{item.analysis}</span>
                          )}
                        </span>
                        <span className="library-item-actions">
                          <button
                            className="library-delete"
                            type="button"
                            title="Edit / move"
                            onClick={(event) => startEdit(item, event)}
                          >
                            <Pencil size={14} aria-hidden="true" />
                          </button>
                          <button
                            className="library-delete"
                            type="button"
                            title="Remove from library"
                            onClick={(event) => deleteLibraryItem(item.id, event)}
                          >
                            <Trash2 size={15} aria-hidden="true" />
                          </button>
                        </span>
                      </div>
                    ),
                  )}
                </div>
              ))}
            </div>
          ) : (
            <p className="library-empty">Analyzed media will appear here.</p>
          )}
        </aside>

        <form className="control-panel" onSubmit={runAnalysis}>
          <div className="brand-row">
            <div className="brand-mark">
              <img src={logoBrutalist} alt="" />
            </div>
            <div>
              <h1>Gemma Media Study</h1>
              <p>Analyze YouTube, local video, and local audio, then chat with Gemma 4.</p>
            </div>
          </div>

          <div className="top-actions">
            <button
              className="setup-button"
              type="button"
              disabled={setupStatus === "running"}
              onClick={runSetup}
            >
              <Settings2 size={18} aria-hidden="true" />
              {setupStatus === "running" ? "Preparing..." : "First-Time Setup"}
            </button>
            <button
              className="refresh-button"
              type="button"
              title="Reload library and Ollama models"
              onClick={() => {
                void refreshLibrary();
                void refreshOllamaModels();
              }}
            >
              <RefreshCw size={18} aria-hidden="true" />
              Refresh
            </button>
          </div>

          {setupStatus !== "idle" && (
            <div className="setup-log" data-status={setupStatus}>
              {setupStatus === "running" && "Preparing the local runtime and models."}
              {setupStatus === "done" && "Local runtime is ready."}
              {setupStatus === "error" && setupLog}
            </div>
          )}

          <details className="help-block">
            <summary>How it works</summary>
            <ul>
              <li>
                <strong>Subtitles:</strong> YouTube captions are used if present;
                otherwise <strong>Whisper</strong> transcribes the speech word-for-word
                (any language, auto-detected).
              </li>
              <li>
                <strong>Analysis &amp; frame summaries:</strong> written in the content's
                own language, or force one with <em>Output Language</em>.
              </li>
              <li>
                <strong>Frames:</strong> N still frames are summarized by content
                (audio files skip frames).
              </li>
              <li>
                <strong>Queue:</strong> add more links anytime — even while analyzing.
                Drag &amp; drop video/audio files too.
              </li>
              <li>
                <strong>Library:</strong> drag items between groups, ✎ rename, 🗑 remove.
              </li>
            </ul>
          </details>

          <label className="field wide">
            <span>
              YouTube URLs or Local File Paths
              {sources.length > 1 ? ` · ${sources.length} items` : ""}
            </span>
            <textarea
              value={url}
              onChange={(event) => setUrl(event.target.value)}
              placeholder={
                "One per line — or drag & drop video/audio files here:\n" +
                "https://www.youtube.com/watch?v=...\n" +
                "C:\\Videos\\lecture.mp4\n" +
                "C:\\Audio\\lesson.mp3"
              }
              rows={4}
              spellCheck={false}
            />
          </label>

          {(batchProgress || queueCount > 0) && (
            <div className="queue-bar">
              <div className="setup-log" data-status="running">
                Processing {batchProgress}
                {queueCount > 0 ? ` · ${queueCount} in queue` : ""}
              </div>
              <button type="button" className="cancel-queue-button" onClick={cancelQueue}>
                Cancel Queue
              </button>
            </div>
          )}

          {batchErrors.length > 0 && (
            <div className="setup-log" data-status="error">
              {batchErrors.length} item(s) failed:
              {batchErrors.map((item) => (
                <div key={item.source} className="batch-error-item">
                  <strong>• {item.source}</strong>
                  <div className="batch-error-msg">{item.error}</div>
                </div>
              ))}
            </div>
          )}

          <div className="field-grid">
            <label className="field">
              <span>Analysis Model</span>
              <select
                value={model}
                onChange={(event) => setModel(event.target.value)}
              >
                {!ollamaModels.some((item) => item.name === model) && (
                  <option value={model}>{model}</option>
                )}
                {ollamaModels.map((item) => (
                  <option value={item.name} key={item.name}>
                    {item.name}
                  </option>
                ))}
              </select>
            </label>

            <label className="field">
              <span>Resolution</span>
              <select
                value={maxHeight}
                onChange={(event) => setMaxHeight(Number(event.target.value))}
              >
                <option value={240}>240p</option>
                <option value={360}>360p</option>
                <option value={480}>480p</option>
                <option value={720}>720p</option>
              </select>
            </label>
          </div>

          <label className="field wide">
            <span>Output Language (analysis &amp; frame summaries)</span>
            <select
              value={outputLanguage}
              onChange={(event) => setOutputLanguage(event.target.value)}
            >
              <option value="auto">Match content (auto)</option>
              <option value="Korean">Korean (한국어)</option>
              <option value="English">English</option>
              <option value="Japanese">Japanese (日本語)</option>
              <option value="Chinese">Chinese (中文)</option>
              <option value="Spanish">Spanish</option>
              <option value="French">French</option>
              <option value="German">German</option>
              <option value="Vietnamese">Vietnamese</option>
            </select>
          </label>

          <div className="field-grid">
            <label className="field">
              <span>Output Tokens</span>
              <input
                type="number"
                min={128}
                max={4096}
                step={128}
                value={maxTokens}
                onChange={(event) => setMaxTokens(Number(event.target.value))}
              />
            </label>

            <label className="field">
              <span>Frames</span>
              <input
                type="number"
                min={0}
                max={120}
                step={2}
                value={frameCount}
                onChange={(event) => setFrameCount(Number(event.target.value))}
              />
            </label>
          </div>

          <label className="toggle-field compact-toggle">
            <input
              type="checkbox"
              checked={keepThinking}
              onChange={(event) => setKeepThinking(event.target.checked)}
            />
            <span>Keep Thinking</span>
          </label>

          <label className="field wide">
            <span>Analysis Prompt</span>
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              rows={7}
            />
          </label>

          <button className="run-button" type="submit" disabled={!canRun}>
            <Play size={18} aria-hidden="true" />
            {status === "running"
              ? sources.length > 0
                ? `Add ${sources.length} to Queue`
                : "Analyzing…"
              : sources.length > 1
                ? `Start Analysis (${sources.length})`
                : "Start Analysis"}
          </button>
        </form>

        <section className="result-panel">
          <div className="view-tabs">
            <button
              type="button"
              data-active={activeView === "study"}
              onClick={() => setActiveView("study")}
            >
              <Film size={17} aria-hidden="true" />
              Media Study
            </button>
            <button
              type="button"
              data-active={activeView === "chat"}
              onClick={() => setActiveView("chat")}
            >
              <MessageSquare size={17} aria-hidden="true" />
              Gemma Chat
            </button>
          </div>

          {activeView === "study" ? (
            <StudyView
              status={status}
              error={error}
              result={result}
              videoSrc={videoSrc}
              mediaRef={mediaRef}
              repeatCue={repeatCue}
              repeatMode={repeatMode}
              playbackRate={playbackRate}
              onRepeatModeChange={setRepeatMode}
              onPlaybackRateChange={setPlaybackRate}
              onClearRepeat={() => setRepeatCue(null)}
              onTimeUpdate={handleTimeUpdate}
              onPlayCue={playCue}
              onOpenFolder={openFolder}
            />
          ) : (
            <ChatView
              endpoint={chatEndpoint}
              model={chatModel}
              input={chatInput}
              status={chatStatus}
              error={chatError}
              messages={visibleChatMessages}
              ollamaModels={ollamaModels}
              ollamaStatus={ollamaStatus}
              onEndpointChange={setChatEndpoint}
              onModelChange={setChatModel}
              onInputChange={setChatInput}
              onRefreshModels={refreshOllamaModels}
              onSubmit={sendChat}
            />
          )}
        </section>
      </section>
    </main>
  );
}

function StudyView(props: {
  status: "idle" | "running" | "done" | "error";
  error: string;
  result: AnalyzeResponse | null;
  videoSrc: string;
  mediaRef: React.RefObject<HTMLMediaElement | null>;
  repeatCue: SubtitleCue | null;
  repeatMode: "loop" | "once";
  playbackRate: number;
  onRepeatModeChange: (mode: "loop" | "once") => void;
  onPlaybackRateChange: (rate: number) => void;
  onClearRepeat: () => void;
  onTimeUpdate: () => void;
  onPlayCue: (cue: SubtitleCue) => void;
  onOpenFolder: (path: string) => void;
}) {
  function changeRate(rate: number) {
    props.onPlaybackRateChange(rate);
    if (props.mediaRef.current) {
      props.mediaRef.current.playbackRate = rate;
    }
  }

  const [copied, setCopied] = useState(false);

  function buildMarkdown(r: AnalyzeResponse): string {
    const lines: string[] = ["# Media Study Notes", "", "## Analysis", "", r.analysis.trim(), ""];
    const frames = r.frames ?? [];
    if (frames.length > 0) {
      lines.push("## Frame Breakdown", "");
      frames.forEach((f, i) => {
        lines.push(`### ${i + 1}. ${formatTime(f.time)}`, "", f.description, "");
      });
    }
    if (r.subtitles.length > 0) {
      lines.push("## Subtitles", "");
      r.subtitles.forEach((c) => {
        lines.push(`- **${formatTime(c.start)}–${formatTime(c.end)}** ${c.text}`);
      });
    }
    return lines.join("\n");
  }

  async function copyMarkdown() {
    if (!props.result) return;
    try {
      await navigator.clipboard.writeText(buildMarkdown(props.result));
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }

  return (
    <>
      <div className="status-strip" data-status={props.status}>
        {props.status === "running" && <Gauge size={19} aria-hidden="true" />}
        {props.status === "done" && <CheckCircle2 size={19} aria-hidden="true" />}
        {props.status === "error" && <AlertCircle size={19} aria-hidden="true" />}
        {props.status === "idle" && <Settings2 size={19} aria-hidden="true" />}
        <span>
          {props.status === "idle" && "Idle"}
          {props.status === "running" && "Loading media and running Gemma analysis"}
          {props.status === "done" && "Analysis complete"}
          {props.status === "error" && "Error"}
        </span>
      </div>

      {props.result ? (
        <div className="result-content">
          <div className="player-layout">
            <section className="video-column">
              {props.result.media_kind === "audio" ? (
                <div className="audio-player-shell">
                  <Film size={34} aria-hidden="true" />
                  <strong>Local Audio Study</strong>
                  <audio
                    ref={(element) => {
                      props.mediaRef.current = element;
                    }}
                    className="audio-player"
                    src={props.videoSrc}
                    controls
                    onTimeUpdate={props.onTimeUpdate}
                  />
                </div>
              ) : (
                <video
                  ref={(element) => {
                    props.mediaRef.current = element;
                  }}
                  className="video-player"
                  src={props.videoSrc}
                  controls
                  onTimeUpdate={props.onTimeUpdate}
                />
              )}
              <div className="file-meta">
                <Film size={18} aria-hidden="true" />
                <div>
                  <strong>{props.result.video_path}</strong>
                  <span>{props.result.study_markdown_path || props.result.analysis_path}</span>
                </div>
                <div className="meta-actions">
                  <button
                    type="button"
                    className="open-folder-button"
                    onClick={() => copyMarkdown()}
                  >
                    <Copy size={16} aria-hidden="true" />
                    {copied ? "Copied!" : "Copy MD"}
                  </button>
                  <button
                    type="button"
                    className="open-folder-button"
                    onClick={() => props.onOpenFolder(props.result!.video_path)}
                  >
                    <FolderOpen size={16} aria-hidden="true" />
                    Open Folder
                  </button>
                </div>
              </div>
              <article className="analysis-output">{props.result.analysis}</article>

              {(props.result.frames ?? []).length > 0 && (
                <section className="frame-gallery">
                  <div className="frame-gallery-header">
                    <strong>Frame Breakdown</strong>
                    <span>{(props.result.frames ?? []).length} frames</span>
                  </div>
                  <div className="frame-gallery-list">
                    {(props.result.frames ?? []).map((frame, index) => (
                      <figure className="frame-card" key={frame.image_path}>
                        <img src={convertFileSrc(frame.image_path)} alt={`Frame ${index + 1}`} />
                        <figcaption>
                          <strong>{formatTime(frame.time)}</strong>
                          <span>{frame.description}</span>
                        </figcaption>
                      </figure>
                    ))}
                  </div>
                </section>
              )}
            </section>

            <aside className="subtitle-rail">
              <div className="subtitle-header">
                <strong>Loop Practice</strong>
                <span>{props.result.subtitles.length} items</span>
              </div>
              <div className="study-tools">
                <div className="segmented-control">
                  <button
                    type="button"
                    data-active={props.repeatMode === "loop"}
                    onClick={() => props.onRepeatModeChange("loop")}
                  >
                    Loop
                  </button>
                  <button
                    type="button"
                    data-active={props.repeatMode === "once"}
                    onClick={() => props.onRepeatModeChange("once")}
                  >
                    Once
                  </button>
                </div>
                <div className="speed-buttons">
                  {[0.75, 1, 1.25].map((rate) => (
                    <button
                      key={rate}
                      type="button"
                      data-active={props.playbackRate === rate}
                      onClick={() => changeRate(rate)}
                    >
                      {rate}x
                    </button>
                  ))}
                </div>
                {props.repeatCue && (
                  <div className="active-cue">
                    <span>Looping {formatTime(props.repeatCue.start)}</span>
                    <button type="button" onClick={props.onClearRepeat}>
                      Clear
                    </button>
                  </div>
                )}
              </div>
              {props.result.subtitles.length > 0 ? (
                <div className="subtitle-list">
                  {props.result.subtitles.map((cue) => (
                    <button
                      key={cue.index}
                      className="subtitle-item"
                      data-active={props.repeatCue?.index === cue.index}
                      type="button"
                      onClick={() => props.onPlayCue(cue)}
                    >
                      <span className="subtitle-time">
                        {formatTime(cue.start)} - {formatTime(cue.end)}
                      </span>
                      <span className="subtitle-ko">{cue.text}</span>
                    </button>
                  ))}
                </div>
              ) : (
                <div className="subtitle-empty">
                  No subtitles were found or generated.
                </div>
              )}
            </aside>
          </div>
          {(props.result.stdout || props.result.stderr) && (
            <details className="logs">
              <summary>Execution Logs</summary>
              <pre>{[props.result.stdout, props.result.stderr].filter(Boolean).join("\n")}</pre>
            </details>
          )}
        </div>
      ) : (
        <div className="empty-state">
          {props.status === "running" ? (
            <div className="processing-scene">
              <div className="processing-ring">
                <Film size={30} aria-hidden="true" />
              </div>
              <strong>Building your media study pack</strong>
              <p>Loading media, reading or generating subtitles, and analyzing with Gemma.</p>
              <div className="process-steps">
                <span>Media</span>
                <span>Subtitles</span>
                <span>Frames</span>
                <span>Analyze</span>
              </div>
              <div className="process-bar" />
            </div>
          ) : props.status === "error" ? (
            <pre>{props.error}</pre>
          ) : (
            <p>Enter a YouTube URL or a local video/audio file path to start studying.</p>
          )}
        </div>
      )}
    </>
  );
}

function ChatView(props: {
  endpoint: string;
  model: string;
  input: string;
  status: "idle" | "running" | "error";
  error: string;
  messages: ChatMessage[];
  ollamaModels: OllamaModel[];
  ollamaStatus: string;
  onEndpointChange: (value: string) => void;
  onModelChange: (value: string) => void;
  onInputChange: (value: string) => void;
  onRefreshModels: () => void;
  onSubmit: (event: FormEvent) => void;
}) {
  return (
    <section className="chat-panel">
      <div className="chat-settings">
        <label className="field">
          <span>Local Gemma Endpoint</span>
          <input
            value={props.endpoint}
            onChange={(event) => props.onEndpointChange(event.target.value)}
            spellCheck={false}
          />
        </label>
        <label className="field">
          <span>Chat Model</span>
          <select
            value={props.model}
            onChange={(event) => props.onModelChange(event.target.value)}
          >
            {props.ollamaModels.length === 0 && (
              <option value={props.model}>{props.model}</option>
            )}
            {props.ollamaModels.map((model) => (
              <option value={model.name} key={model.name}>
                {model.name}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="ollama-row">
        <button type="button" onClick={props.onRefreshModels}>
          Check Ollama Models
        </button>
        <span>{props.ollamaStatus || "Ollama model list not checked yet."}</span>
      </div>

      <div className="chat-messages">
        {props.messages.length === 0 ? (
          <div className="chat-empty">
            Ask Gemma 4 about the media, subtitles, and study questions.
          </div>
        ) : (
          props.messages.map((message, index) => (
            <div className="chat-message" data-role={message.role} key={index}>
              <strong>{message.role === "user" ? "You" : "Gemma"}</strong>
              <p>{message.content}</p>
            </div>
          ))
        )}
        {props.status === "running" && (
          <div className="chat-message" data-role="assistant">
            <strong>Gemma</strong>
            <p>Generating response...</p>
          </div>
        )}
        {props.status === "error" && <pre className="chat-error">{props.error}</pre>}
      </div>

      <form className="chat-input-row" onSubmit={props.onSubmit}>
        <textarea
          value={props.input}
          onChange={(event) => props.onInputChange(event.target.value)}
          placeholder="Ask Gemma 4"
          rows={3}
        />
        <button type="submit" disabled={props.status === "running" || !props.input.trim()}>
          <Send size={18} aria-hidden="true" />
        </button>
      </form>
    </section>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
