//! Integration tests for melops CLI.

use clap::Parser;
use melops::cli::{Cli, run};

const URL: &str = "https://youtu.be/jNQXAC9IVRw";

#[tokio::test(flavor = "current_thread")]
#[ignore = "network I/O and model download required"]
async fn dl_downloads_and_transcribes() {
    let temp_dir = std::env::temp_dir().join("melops").join("test");

    // Clean up previous test run
    if temp_dir.exists() {
        std::fs::remove_dir_all(&temp_dir).ok();
    }
    std::fs::create_dir_all(&temp_dir).expect("failed to create temp dir");

    let cli = Cli::parse_from(["mel", "dl", URL, "-o", temp_dir.to_str().unwrap()]);

    run(cli).await.expect("failed to download and transcribe");

    // Verify SRT file was created
    // Expected path: temp_dir/Youtube/jawed/jNQXAC9IVRw/Me_at_the_zoo.srt
    // Note: restrictfilenames=true sanitizes title (spaces -> underscores)
    let srt_path = temp_dir.join("Youtube/jawed/jNQXAC9IVRw/Me_at_the_zoo.srt");

    assert!(
        srt_path.exists(),
        "SRT file not found: {:?}",
        srt_path.display()
    );
}
