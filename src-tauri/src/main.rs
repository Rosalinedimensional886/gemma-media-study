use serde::{Deserialize, Serialize};
use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Debug, Deserialize)]
struct AnalyzeRequest {
    url: String,
    model: String,
    max_height: u32,
    max_new_tokens: u32,
    prompt: String,
    keep_thinking: bool,
    #[serde(default)]
    frame_count: Option<u32>,
    #[serde(default)]
    output_language: String,
}

#[derive(Debug, Serialize)]
struct AnalyzeResponse {
    analysis: String,
    analysis_path: String,
    study_markdown_path: Option<String>,
    frames_markdown_path: Option<String>,
    video_path: String,
    media_kind: Option<String>,
    subtitle_path: Option<String>,
    thumbnail_path: Option<String>,
    subtitles: Vec<SubtitleCue>,
    frames: Vec<FrameNote>,
    stdout: String,
    stderr: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct SummaryFile {
    analysis: String,
    analysis_path: String,
    study_markdown_path: Option<String>,
    #[serde(default)]
    frames_markdown_path: Option<String>,
    video_path: String,
    media_kind: Option<String>,
    subtitle_path: Option<String>,
    thumbnail_path: Option<String>,
    subtitles: Vec<SubtitleCue>,
    #[serde(default)]
    frames: Vec<FrameNote>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct SubtitleCue {
    index: usize,
    start: f64,
    end: f64,
    text: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct FrameNote {
    image_path: String,
    time: f64,
    description: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct LibraryItem {
    id: String,
    title: String,
    #[serde(default)]
    group: String,
    created_at: u128,
    analysis: String,
    analysis_path: String,
    study_markdown_path: Option<String>,
    #[serde(default)]
    frames_markdown_path: Option<String>,
    video_path: String,
    media_kind: Option<String>,
    subtitle_path: Option<String>,
    thumbnail_path: Option<String>,
    subtitles: Vec<SubtitleCue>,
    #[serde(default)]
    frames: Vec<FrameNote>,
}

#[derive(Debug, Deserialize)]
struct SetupRequest {
    gemma_model: String,
}

#[derive(Debug, Serialize)]
struct SetupResponse {
    stdout: String,
    stderr: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    endpoint: String,
    api_key: String,
    model: String,
    messages: Vec<ChatMessage>,
}

#[derive(Debug, Serialize)]
struct ChatResponse {
    message: ChatMessage,
}

#[derive(Debug, Serialize)]
struct OllamaModel {
    name: String,
}

#[derive(Debug, Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaTagModel>,
}

#[derive(Debug, Deserialize)]
struct OllamaTagModel {
    name: String,
}

#[derive(Debug, Serialize)]
struct OpenAiChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct OpenAiChatResponse {
    choices: Vec<OpenAiChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: ChatMessage,
}

#[tauri::command]
async fn analyze_youtube(request: AnalyzeRequest) -> Result<AnalyzeResponse, String> {
    tauri::async_runtime::spawn_blocking(move || run_python_analyzer(request))
        .await
        .map_err(|err| format!("Failed to join analysis task: {err}"))?
}

#[tauri::command]
async fn setup_environment(request: SetupRequest) -> Result<SetupResponse, String> {
    tauri::async_runtime::spawn_blocking(move || run_setup(request))
        .await
        .map_err(|err| format!("Failed to join setup task: {err}"))?
}

#[tauri::command]
async fn chat_gemma(request: ChatRequest) -> Result<ChatResponse, String> {
    tauri::async_runtime::spawn_blocking(move || run_chat(request))
        .await
        .map_err(|err| format!("Failed to join chat task: {err}"))?
}

#[tauri::command]
async fn list_ollama_models() -> Result<Vec<OllamaModel>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let client = reqwest::blocking::Client::new();
        let response = client
            .get("http://127.0.0.1:11434/api/tags")
            .send()
            .map_err(|err| format!("Failed to connect to Ollama at 127.0.0.1:11434: {err}"))?;
        let status = response.status();
        let text = response
            .text()
            .map_err(|err| format!("Failed to read Ollama model list: {err}"))?;
        if !status.is_success() {
            return Err(format!("Ollama returned {status}.\n\n{text}"));
        }
        let parsed: OllamaTagsResponse = serde_json::from_str(&text)
            .map_err(|err| format!("Failed to parse Ollama model list: {err}\n\n{text}"))?;
        Ok(parsed
            .models
            .into_iter()
            .map(|model| OllamaModel { name: model.name })
            .collect())
    })
    .await
    .map_err(|err| format!("Failed to join Ollama model task: {err}"))?
}

#[tauri::command]
async fn open_path(path: String) -> Result<(), String> {
    // Open a file's containing folder (or a folder directly) in Windows Explorer.
    let target = PathBuf::from(&path);
    let folder = if target.is_dir() {
        target
    } else {
        target
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or(target)
    };
    if !folder.exists() {
        return Err(format!("Path does not exist: {}", folder.display()));
    }
    Command::new("explorer")
        .arg(&folder)
        .spawn()
        .map_err(|err| format!("Failed to open folder {}: {err}", folder.display()))?;
    Ok(())
}

#[tauri::command]
async fn list_library() -> Result<Vec<LibraryItem>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let project_dir = project_dir()?;
        read_library(&project_dir)
    })
    .await
    .map_err(|err| format!("Failed to join library task: {err}"))?
}

#[tauri::command]
async fn load_library_item(id: String) -> Result<AnalyzeResponse, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let project_dir = project_dir()?;
        let library = read_library(&project_dir)?;
        let item = library
            .into_iter()
            .find(|item| item.id == id)
            .ok_or_else(|| "Library item not found.".to_string())?;
        Ok(AnalyzeResponse {
            analysis: item.analysis,
            analysis_path: item.analysis_path,
            study_markdown_path: item.study_markdown_path,
            frames_markdown_path: item.frames_markdown_path,
            video_path: item.video_path,
            media_kind: item.media_kind,
            subtitle_path: item.subtitle_path,
            thumbnail_path: item.thumbnail_path,
            subtitles: item.subtitles,
            frames: item.frames,
            stdout: String::new(),
            stderr: String::new(),
        })
    })
    .await
    .map_err(|err| format!("Failed to join library load task: {err}"))?
}

#[tauri::command]
async fn update_library_item(
    id: String,
    title: String,
    group: String,
) -> Result<Vec<LibraryItem>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let project_dir = project_dir()?;
        let mut items = read_library(&project_dir)?;
        let mut found = false;
        for item in items.iter_mut() {
            if item.id == id {
                let trimmed = title.trim();
                if !trimmed.is_empty() {
                    item.title = trimmed.to_string();
                }
                item.group = group.trim().to_string();
                found = true;
                break;
            }
        }
        if !found {
            return Err("Library item not found.".to_string());
        }
        write_library(&project_dir, &items)?;
        Ok(items)
    })
    .await
    .map_err(|err| format!("Failed to join library update task: {err}"))?
}

#[tauri::command]
async fn delete_library_item(id: String) -> Result<Vec<LibraryItem>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let project_dir = project_dir()?;
        let mut items = read_library(&project_dir)?;
        let before = items.len();
        items.retain(|item| item.id != id);
        if items.len() == before {
            return Err("Library item not found.".to_string());
        }
        // Only the library entry is removed; the downloaded files stay on disk.
        write_library(&project_dir, &items)?;
        Ok(items)
    })
    .await
    .map_err(|err| format!("Failed to join library delete task: {err}"))?
}

fn run_python_analyzer(request: AnalyzeRequest) -> Result<AnalyzeResponse, String> {
    let source = request.url.trim();
    if source.is_empty() {
        return Err("YouTube URL or local media path is required.".to_string());
    }

    let project_dir = project_dir()?;
    let script_path = project_dir.join("analyze_youtube_gemma.py");
    if !script_path.exists() {
        return Err(format!("Python script not found: {}", script_path.display()));
    }

    let python = python_executable(&project_dir).ok_or_else(|| {
        format!(
            "Python environment is not set up. The virtualenv was not found at {}.\n\n\
             Click \"First-Time Setup\" first: it creates .venv, installs the media \
             runtime (yt-dlp, PyAV, faster-whisper), and pulls the Ollama model.",
            project_dir
                .join(".venv")
                .join("Scripts")
                .join("python.exe")
                .display()
        )
    })?;
    let summary_path = project_dir
        .join("downloads")
        .join(format!("tauri-summary-{}.json", timestamp_millis()));
    let mut command = Command::new(&python);
    command
        .current_dir(&project_dir)
        .arg(&script_path)
        .arg("--model")
        .arg(request.model.trim())
        .arg("--max-height")
        .arg(request.max_height.to_string())
        .arg("--max-new-tokens")
        .arg(request.max_new_tokens.to_string())
        .arg("--prompt")
        .arg(request.prompt)
        .arg("--json-summary-path")
        .arg(&summary_path);

    if let Some(frame_count) = request.frame_count {
        command.arg("--frame-count").arg(frame_count.to_string());
    }

    let output_language = request.output_language.trim();
    if !output_language.is_empty() {
        command.arg("--output-language").arg(output_language);
    }

    if is_remote_source(source) {
        command.arg(source);
    } else {
        command.arg("--local-media").arg(source);
    }

    if request.keep_thinking {
        command.arg("--keep-thinking");
    }

    let output = command
        .output()
        .map_err(|err| format!("Failed to run Python analyzer with {}: {err}", python.display()))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if !output.status.success() {
        return Err(format!(
            "Analyzer failed with status {}.\n\nSTDOUT:\n{}\n\nSTDERR:\n{}",
            output.status, stdout, stderr
        ));
    }

    let summary_text = std::fs::read_to_string(&summary_path)
        .map_err(|err| format!("Failed to read summary file {}: {err}", summary_path.display()))?;
    // The summary file is a one-shot hand-off from Python; its contents now live in
    // library.json, so remove it instead of letting these temp files pile up in downloads/.
    let _ = std::fs::remove_file(&summary_path);
    let summary: SummaryFile = serde_json::from_str(&summary_text)
        .map_err(|err| format!("Failed to parse summary file {}: {err}", summary_path.display()))?;
    save_library_item(&project_dir, &summary)?;

    Ok(AnalyzeResponse {
        analysis: summary.analysis,
        analysis_path: summary.analysis_path,
        study_markdown_path: summary.study_markdown_path,
        frames_markdown_path: summary.frames_markdown_path,
        video_path: summary.video_path,
        media_kind: summary.media_kind,
        subtitle_path: summary.subtitle_path,
        thumbnail_path: summary.thumbnail_path,
        subtitles: summary.subtitles,
        frames: summary.frames,
        stdout,
        stderr,
    })
}

fn library_path(project_dir: &Path) -> PathBuf {
    project_dir.join("downloads").join("library.json")
}

fn read_library(project_dir: &Path) -> Result<Vec<LibraryItem>, String> {
    let path = library_path(project_dir);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let text = std::fs::read_to_string(&path)
        .map_err(|err| format!("Failed to read library {}: {err}", path.display()))?;
    serde_json::from_str(&text)
        .map_err(|err| format!("Failed to parse library {}: {err}", path.display()))
}

fn write_library(project_dir: &Path, items: &[LibraryItem]) -> Result<(), String> {
    let path = library_path(project_dir);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|err| format!("Failed to create library directory {}: {err}", parent.display()))?;
    }
    let text = serde_json::to_string_pretty(items)
        .map_err(|err| format!("Failed to serialize library: {err}"))?;
    std::fs::write(&path, text)
        .map_err(|err| format!("Failed to write library {}: {err}", path.display()))
}

fn save_library_item(project_dir: &Path, summary: &SummaryFile) -> Result<(), String> {
    let mut items = read_library(project_dir)?;
    let id = stable_media_id(&summary.video_path);
    let title = Path::new(&summary.video_path)
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("Untitled media")
        .to_string();
    // Preserve a user-set group/title if this media was analyzed before.
    let existing = items.iter().find(|it| it.id == id);
    let group = existing.map(|it| it.group.clone()).unwrap_or_default();
    let title = existing
        .map(|it| it.title.clone())
        .filter(|t| !t.is_empty())
        .unwrap_or(title);
    let item = LibraryItem {
        id: id.clone(),
        title,
        group,
        created_at: timestamp_millis(),
        analysis: summary.analysis.clone(),
        analysis_path: summary.analysis_path.clone(),
        study_markdown_path: summary.study_markdown_path.clone(),
        frames_markdown_path: summary.frames_markdown_path.clone(),
        video_path: summary.video_path.clone(),
        media_kind: summary.media_kind.clone(),
        subtitle_path: summary.subtitle_path.clone(),
        thumbnail_path: summary.thumbnail_path.clone(),
        subtitles: summary.subtitles.clone(),
        frames: summary.frames.clone(),
    };

    items.retain(|existing| existing.id != id);
    items.insert(0, item);
    write_library(project_dir, &items)
}

fn stable_media_id(path: &str) -> String {
    path.chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

fn run_chat(request: ChatRequest) -> Result<ChatResponse, String> {
    if request.endpoint.trim().is_empty() {
        return Err("Local LLM endpoint is required.".to_string());
    }
    if request.model.trim().is_empty() {
        return Err("Chat model is required.".to_string());
    }

    let client = reqwest::blocking::Client::new();
    let body = OpenAiChatRequest {
        model: request.model.trim().to_string(),
        messages: request.messages,
        stream: false,
    };

    let mut builder = client.post(request.endpoint.trim()).json(&body);
    if !request.api_key.trim().is_empty() {
        builder = builder.bearer_auth(request.api_key.trim());
    }

    let response = builder
        .send()
        .map_err(|err| format!("Failed to call local Gemma endpoint: {err}"))?;
    let status = response.status();
    let text = response
        .text()
        .map_err(|err| format!("Failed to read chat response: {err}"))?;

    if !status.is_success() {
        return Err(format!("Local Gemma endpoint returned {status}.\n\n{text}"));
    }

    let parsed: OpenAiChatResponse = serde_json::from_str(&text)
        .map_err(|err| format!("Failed to parse OpenAI-compatible chat response: {err}\n\n{text}"))?;
    let message = parsed
        .choices
        .into_iter()
        .next()
        .map(|choice| choice.message)
        .ok_or_else(|| "Chat response did not include any choices.".to_string())?;

    Ok(ChatResponse { message })
}

fn run_setup(request: SetupRequest) -> Result<SetupResponse, String> {
    let project_dir = project_dir()?;
    let python = PathBuf::from("python");
    let venv_dir = project_dir.join(".venv");
    let venv_python = venv_dir.join("Scripts").join("python.exe");

    let mut stdout = String::new();
    let mut stderr = String::new();

    if !venv_python.exists() {
        let output = Command::new(&python)
            .current_dir(&project_dir)
            .arg("-m")
            .arg("venv")
            .arg(&venv_dir)
            .output()
            .map_err(|err| format!("Failed to create virtualenv: {err}"))?;
        append_output(&mut stdout, &mut stderr, output)?;
    }

    for args in [
        vec!["-m", "pip", "install", "--upgrade", "pip"],
        vec!["-m", "pip", "install", "-r", "requirements.txt"],
    ] {
        let output = Command::new(&venv_python)
            .current_dir(&project_dir)
            .args(args)
            .output()
            .map_err(|err| format!("Failed to run pip with {}: {err}", venv_python.display()))?;
        append_output(&mut stdout, &mut stderr, output)?;
    }

    let predownload = project_dir.join("predownload_models.py");
    let mut model_args = vec![predownload.to_string_lossy().to_string()];
    let gemma_model = request.gemma_model.trim();
    if !gemma_model.is_empty() {
        model_args.push("--model".to_string());
        model_args.push(gemma_model.to_string());
    }

    if model_args.len() > 1 {
        let output = Command::new(&venv_python)
            .current_dir(&project_dir)
            .args(model_args)
            .output()
            .map_err(|err| format!("Failed to predownload models: {err}"))?;
        append_output(&mut stdout, &mut stderr, output)?;
    }

    Ok(SetupResponse { stdout, stderr })
}

fn project_dir() -> Result<PathBuf, String> {
    let mut starts: Vec<PathBuf> = Vec::new();
    if let Ok(current) = env::current_dir() {
        starts.push(current);
    }
    if let Ok(exe) = env::current_exe() {
        if let Some(parent) = exe.parent() {
            starts.push(parent.to_path_buf());
        }
    }

    for start in starts {
        for candidate in start.ancestors() {
            if candidate.join("analyze_youtube_gemma.py").exists() {
                return Ok(candidate.to_path_buf());
            }
        }
    }

    Err("Failed to resolve project directory containing analyze_youtube_gemma.py".to_string())
}

fn python_executable(project_dir: &Path) -> Option<PathBuf> {
    // Only use the project virtualenv. Falling back to a system "python" hides the
    // real problem (missing GPU deps) behind a confusing ModuleNotFoundError, so we
    // return None and let the caller tell the user to run First-Time Setup.
    let venv_python = project_dir.join(".venv").join("Scripts").join("python.exe");
    if venv_python.exists() {
        Some(venv_python)
    } else {
        None
    }
}

fn is_remote_source(source: &str) -> bool {
    source.starts_with("http://") || source.starts_with("https://")
}

fn append_output(
    stdout: &mut String,
    stderr: &mut String,
    output: std::process::Output,
) -> Result<(), String> {
    stdout.push_str(&String::from_utf8_lossy(&output.stdout));
    stderr.push_str(&String::from_utf8_lossy(&output.stderr));
    if !output.status.success() {
        return Err(format!(
            "Command failed with status {}.\n\nSTDOUT:\n{}\n\nSTDERR:\n{}",
            output.status, stdout, stderr
        ));
    }
    Ok(())
}

fn timestamp_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .invoke_handler(tauri::generate_handler![
            analyze_youtube,
            setup_environment,
            chat_gemma,
            list_ollama_models,
            list_library,
            load_library_item,
            update_library_item,
            delete_library_item,
            open_path
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
