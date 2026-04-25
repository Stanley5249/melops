//! ASR preset download integration tests.
//!
//! Tests: YouTube download, path grouping (Extractor/uploader/id).
//!
//! Uses "Me at the zoo" (jNQXAC9IVRw) - predictable metadata.

use eyre::{Context, OptionExt, Result, ensure};
use melops_dl::dl::download;
use melops_dl::info::DownloadInfo;
use melops_dl::params::{ASR_OUTPUT_TEMPLATE, DownloadParams, OutputPaths, OutputTemplates};
use std::fs::{create_dir_all, remove_dir_all};
use std::path::PathBuf;
use std::sync::LazyLock;
use tokio::sync::broadcast;

/// Broadcast channel buffer size for test downloads.
const DOWNLOAD_PATH_CHANNEL_SIZE: usize = 10;

const TEST_URL: &str = "https://youtu.be/jNQXAC9IVRw";
const TEST_ID: &str = "jNQXAC9IVRw";
const TEST_REL_DIR: &str = "Youtube/jawed/jNQXAC9IVRw";

struct TestContext {
    file_path: PathBuf,
    info: DownloadInfo,
}

static TEST_CONTEXT: LazyLock<Result<TestContext>> = LazyLock::new(|| {
    let temp_dir = create_temp_dir();

    let mut preset = DownloadParams::asr();
    preset.paths = Some(OutputPaths::simple(&temp_dir, &temp_dir));
    preset.outtmpl = Some(OutputTemplates::simple(ASR_OUTPUT_TEMPLATE.to_string()));

    let (tx, mut rx) = broadcast::channel(DOWNLOAD_PATH_CHANNEL_SIZE);

    let info = download(TEST_URL, preset, tx).context("yt-dlp download failed for ASR preset")?;

    // Collect file paths from broadcast channel (blocking receive in sync context)
    let mut audio_paths = Vec::new();
    while let Ok(path) = rx.try_recv() {
        audio_paths.push(path);
    }

    let file_path = audio_paths
        .first()
        .ok_or_eyre("download did not return any file paths")?
        .clone();

    ensure!(
        file_path.exists(),
        "downloaded file not found at: {:?}",
        file_path.display()
    );

    Ok(TestContext { file_path, info })
});

fn create_temp_dir() -> PathBuf {
    let mut temp_dir = std::env::temp_dir();
    temp_dir.push("melops");
    temp_dir.push("test");

    if temp_dir.exists() {
        remove_dir_all(&temp_dir).ok();
    }

    create_dir_all(&temp_dir).expect("failed to create temp dir");

    temp_dir
}

#[track_caller]
fn get_test_context() -> &'static TestContext {
    TEST_CONTEXT.as_ref().expect("download failed")
}

#[test]
#[ignore = "network I/O"]
fn audio_file_exists() {
    let ctx = get_test_context();

    assert!(
        ctx.file_path.exists(),
        "audio file not found: {:?}",
        ctx.file_path.display()
    );
}

#[test]
#[ignore = "network I/O"]
fn info_file_exists() {
    let ctx = get_test_context();
    let info_file = ctx.file_path.with_extension("info.json");

    assert!(
        info_file.exists(),
        "info.json not found: {:?}",
        info_file.display()
    );
}

#[test]
#[ignore = "network I/O"]
fn path_structure() {
    let ctx = get_test_context();
    let parent = ctx.file_path.parent().expect("file has no parent");

    assert!(
        parent.ends_with(TEST_REL_DIR),
        "expected path under {TEST_REL_DIR}, got {:?}",
        ctx.file_path.display()
    );
}

#[test]
#[ignore = "network I/O"]
fn info_dict_fields() {
    let ctx = get_test_context();

    // Only `id` is guaranteed stable in DownloadInfo; other fields (title, uploader, etc.)
    // can vary by extractor version and are available in the .info.json file.
    assert_eq!(
        ctx.info.id, TEST_ID,
        "expected video ID to be {TEST_ID}, got {}",
        ctx.info.id
    );
}
