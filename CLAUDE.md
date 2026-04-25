# Development Rules

## Terminology

- **ASR** (Automatic Speech Recognition) - Raw algorithm listening to audio, identifying words.
- **STT** (Speech-to-Text) - Wraps ASR with punctuation, speaker identification, formatting.
- **Captioning** - STT output synchronized with timeline for video (e.g., SRT).

## Data Flow

- _File input_: file → ffmpeg decode → chunked f32 stream → TDT inference → detokenization → subtitle regrouping (DP) → SRT output
- _URL input_: URL → yt-dlp download → ffmpeg decode → chunked f32 stream → TDT inference → SRT output

## Build System

Use `pixi run` for all commands. Read `pixi.toml` for available tasks, features, and environments.

### Runtime Fallback Order

`melops/src/ort.rs::init()` tries providers in order: CUDA → TensorRT → OpenVINO → DirectML → CoreML → CPU.

### Cache Architecture

Three cache directories (`.cache/melops/`):

1. **models/** - Downloaded ONNX models from HuggingFace Hub (via `hf-hub`)
2. **transcriptions/** - Cached segments keyed by audio content hash (via `cacache`)
3. **ort/** - OpenVINO compilation cache (IR files, faster reloads)

**Cache strategy** (`--cache-strategy`): `auto` (default), `use` (error if missing), `force` (recompute).

## CLI Design Philosophy

### State Layers

Three distinct concerns:

1. **CLI State** - Raw clap arguments
2. **Validated State** - Type-safe configurations
3. **Application State** - Runtime objects that can fail

```rust
pub fn run(command: CapCommand) -> Result<()> {
    let config = CapConfig::try_from(command.args)?;   // CLI → Validated
    let cache = Cache::load(CacheConfig::from(command.cache_args))?;  // Validated → App
    let model = config.model.load()?;
    caption(&config, &model, &mut cache)
}
```

Validate early before expensive operations. Each domain function has single responsibility with dependencies passed in. Cache owns refresh flags — no separate refresh parameters.

## ONNX Model Structure

### Blank Token Handling (Parakeet-TDT-0.6b-v3)

- NeMo vocab: 8192 tokens (IDs 0–8191); decoder output: 8198 logits = 8192 vocab + 1 blank + 5 duration
- `blank_id = vocab_size = 8192` — no `<blk>` string token in tokenizer.json or vocab.txt
- Decoder filters blanks during greedy search; detokenizer never receives blank tokens

## Cargo.toml

Never add dependencies manually. Use `cargo add`, `cargo update`.

```bash
pixi run cargo add pyo3@0.27
pixi run cargo update -p pyo3
```

## Rust Format Style

### Derive Ordering

Single line, sorted: std traits alphabetically, then 3rd party alphabetically.

```rust
#[derive(Clone, Debug, Default, IntoPyObject)]  // ✓ Clone, Debug, Default (std) → IntoPyObject (pyo3)
```

## Testing

```rust
#[test]
#[ignore = "network I/O"]
fn downloads_from_youtube() { }
```

## Error Handling

Use `eyre` in libraries; `color-eyre` in binaries/tests for colored output.

## Logging

- **stdout**: data output only (SRT subtitles, JSON results)
- **stderr**: diagnostics, progress, timing (via `tracing`)

```rust
tracing::debug!(current = i, total, "processing chunk");
tracing::info!(%duration, "model loaded");
tracing::info!(path = ?audio_path.display(), "transcribing audio");
```

Use `%` for Display, `?` for Debug. Debug format for paths enables IDE clickability (wraps in quotes).

## PyO3 0.27

```rust
pub struct Args {
    #[pyo3(item("ffmpeg"))]  // Maps to Python key "ffmpeg"
    pub key: Vec<String>,
}
```

## Commit Style

Angular convention. Scale matches change size.

Format: `<type>(<scope>): <subject>` — header ≤50 chars, body wrap at 72, imperative mood.
Types: `feat`, `fix`, `refactor`, `style`, `docs`, `test`, `chore`
Scopes: `cli`, `dl`, `parakeet`

Explain **why** over **what** — reason for change, behavior users notice, problem being solved.

## Before Committing

```bash
pixi run -e openvino cargo test --workspace
pixi run -e openvino cargo clippy --fix --workspace --allow-dirty
pixi run -e openvino cargo clippy --workspace
```
