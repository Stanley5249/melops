//! ASR preset download integration tests.
//!
//! Tests: YouTube download, WAV format (16kHz mono 16-bit PCM via hound),
//! path grouping (Extractor/uploader/id).
//!
//! Uses "Me at the zoo" (jNQXAC9IVRw) - predictable metadata.

use eyre::{ensure, Context, OptionExt, Result};
use melops_dl::asr::{AudioFormat, ASR_OUTPUT_TEMPLATE};
use melops_dl::dl::download;
use melops_dl::info::DownloadInfo;
use melops_dl::params::{DownloadParams, OutputPaths, OutputTemplates};
use std::fs::{create_dir_all, remove_dir_all};
use std::path::PathBuf;
use std::sync::LazyLock;
use tokio::sync::broadcast;

/// Broadcast channel buffer size for test downloads.
const DOWNLOAD_PATH_CHANNEL_SIZE: usize = 10;

const TEST_URL: &str = "https://youtu.be/jNQXAC9IVRw";
const TEST_ID: &str = "jNQXAC9IVRw";
const TEST_REL_PATH: &str = "Youtube/jawed/jNQXAC9IVRw/Me_at_the_zoo.wav";

struct TestContext {
    file_path: PathBuf,
    info: DownloadInfo,
}

static TEST_CONTEXT: LazyLock<Result<TestContext>> = LazyLock::new(|| {
    let temp_dir = create_temp_dir();

    let mut preset: DownloadParams = AudioFormat::Pcm16.into();
    preset.paths = Some(OutputPaths::simple(&temp_dir, &temp_dir));
    preset.outtmpl = Some(OutputTemplates::simple(ASR_OUTPUT_TEMPLATE.to_string()));

    // Create broadcast channel for receiving file paths
    let (tx, mut rx) = broadcast::channel(DOWNLOAD_PATH_CHANNEL_SIZE);

    let info =
        download(TEST_URL, preset, tx).context("yt-dlp download failed for ASR Pcm16 preset")?;

    // Collect file paths from broadcast channel (blocking receive in sync context)
    let mut audio_paths = Vec::new();
    while let Ok(path) = rx.try_recv() {
        audio_paths.push(path);
    }

    // Validate file_path was returned and exists
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

    // Clean up previous test run
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
fn wav_file_exist() {
    let ctx = get_test_context();

    assert!(
        ctx.file_path.exists(),
        "WAV file not found: {:?}",
        ctx.file_path.display()
    );
}

#[test]
#[ignore = "network I/O"]
fn info_file_exist() {
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

    assert!(
        ctx.file_path.ends_with(TEST_REL_PATH),
        "expected path to end with {TEST_REL_PATH}, got {:?}",
        ctx.file_path.display()
    );
}

#[test]
#[ignore = "network I/O"]
fn wav_format() {
    let ctx = get_test_context();

    let reader = hound::WavReader::open(&ctx.file_path).expect("failed to open WAV file");
    let spec = reader.spec();

    assert!(
        matches!(
            spec,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            }
        ),
        "unexpected WAV format: {spec:?}, expected 16kHz mono 16-bit PCM"
    );
}

#[test]
#[ignore = "network I/O"]
fn info_dict_fields() {
    let ctx = get_test_context();

    // Only `id` is guaranteed to be present in DownloadInfo.
    // Other fields (title, extractor_key, uploader, etc.) can be accessed
    // from the .info.json file if needed.
    assert_eq!(
        ctx.info.id, TEST_ID,
        "expected video ID to be {TEST_ID}, got {}",
        ctx.info.id
    );
}
