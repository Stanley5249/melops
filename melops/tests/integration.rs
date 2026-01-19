//! Integration tests for melops CLI.

use clap::Parser;
use melops::cache::CacheDir;
use melops::cli::{Cli, run};

const URL: &str = "https://youtu.be/jNQXAC9IVRw";
const MODEL_ID: &str = "nvidia--parakeet-tdt-0.6b-v3";

#[tokio::test(flavor = "current_thread")]
#[ignore = "network I/O and model download required"]
async fn mel_dl() {
    color_eyre::install().expect("failed to install color_eyre");

    let _guard = melops::tracing::init().expect("failed to initialize tracing");

    melops::ort::init().expect("failed to initialize ort");

    let temp_dir = std::env::temp_dir().join("melops").join("test");

    // Clean up previous test run
    std::fs::remove_dir_all(&temp_dir).ok();

    std::fs::create_dir_all(&temp_dir).expect("failed to create temp dir");

    // Get model path from cache
    let cache_dir = CacheDir::new(None).expect("failed to get cache dir");

    let model_path = cache_dir.model(MODEL_ID);

    // Verify model exists in cache
    assert!(
        model_path.exists(),
        "model not found in cache: {:?}\nRun export script first to populate cache",
        model_path.display()
    );

    let cli = Cli::parse_from([
        "mel",
        "dl",
        "--model-id",
        model_path.to_str().unwrap(),
        URL,
        "-o",
        temp_dir.to_str().unwrap(),
    ]);

    run(cli).await.expect("failed to download and transcribe");

    // Verify SRT file was created
    // Expected path: <temp_dir>/Youtube/jawed/jNQXAC9IVRw/Me_at_the_zoo.srt
    // Note: restrictfilenames=true sanitizes title (spaces -> underscores)
    let srt_path = temp_dir.join("Youtube/jawed/jNQXAC9IVRw/Me_at_the_zoo.srt");

    assert!(
        srt_path.exists(),
        "SRT file not found: {:?}",
        srt_path.display()
    );
}
