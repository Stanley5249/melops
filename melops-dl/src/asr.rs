//! ASR audio presets: 16kHz mono WAV extraction.
//!
//! **Formats:** [`AudioFormat::Pcm16`] (16-bit, standard), [`AudioFormat::Float32`] (32-bit, higher precision)
//!
//! ```no_run
//! use melops_dl::{dl::download, asr::AudioFormat};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let (file_path, info) = download("https://youtube.com/watch?v=example", AudioFormat::Pcm16.into())?;
//! println!("Downloaded '{}' to {:?}", info.title, file_path);
//! # Ok(())
//! # }
//! ```
//!
//! **Output:** `downloads/Extractor/uploader/id/title.wav` + `title.info.json`

use crate::params::{
    DownloadParams, OutputPaths, OutputTemplates, PostProcessor, PostProcessorArgs,
};

/// Output template for ASR: "<Extractor>/<uploader>/<id>/<title>.<ext>"
pub const ASR_OUTPUT_TEMPLATE: &str = "%(extractor_key)s/%(uploader)s/%(id)s/%(title)s.%(ext)s";

/// 16kHz mono WAV format (`pcm_s16le` or `pcm_f32le`).
#[derive(Copy, Clone, Debug, Default)]
pub enum AudioFormat {
    /// 16-bit PCM (standard, smaller files)
    #[default]
    Pcm16,
    /// 32-bit float PCM (higher precision, ~2x larger)
    Float32,
}

impl From<AudioFormat> for PostProcessorArgs {
    /// FFmpeg args: `-ar 16000 -ac 1 -c:a pcm_s16le` (Pcm16) or `pcm_f32le` (Float32)
    fn from(format: AudioFormat) -> Self {
        let ffmpeg = match format {
            AudioFormat::Pcm16 => vec![
                "-ar".to_string(),
                "16000".to_string(), // 16kHz sample rate
                "-ac".to_string(),
                "1".to_string(), // mono
                "-c:a".to_string(),
                "pcm_s16le".to_string(), // 16-bit PCM
            ],
            AudioFormat::Float32 => vec![
                "-ar".to_string(),
                "16000".to_string(), // 16kHz sample rate
                "-ac".to_string(),
                "1".to_string(), // mono
                "-c:a".to_string(),
                "pcm_f32le".to_string(), // 32-bit float PCM
            ],
        };
        Self { ffmpeg }
    }
}

impl From<AudioFormat> for DownloadParams {
    /// ASR preset: best audio → 16kHz mono WAV, organized by `Extractor/uploader/id`, saves `.info.json`
    fn from(format: AudioFormat) -> Self {
        Self {
            format: Some("ba".to_string()),
            paths: Some(OutputPaths::system_default()),
            outtmpl: Some(OutputTemplates::simple(ASR_OUTPUT_TEMPLATE.to_string())),
            postprocessors: Some(vec![PostProcessor {
                key: "FFmpegExtractAudio".to_string(),
                preferredcodec: Some("wav".to_string()),
            }]),
            postprocessor_args: Some(format.into()),
            writeinfojson: Some(true),
            restrictfilenames: Some(true),
            getcomments: None,
            quiet: None,
            no_warnings: None,
            keepvideo: Some(true),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn postprocessor_args_pcm16() {
        let args: PostProcessorArgs = AudioFormat::Pcm16.into();
        assert_eq!(
            args.ffmpeg,
            ["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"]
        );
    }

    #[test]
    fn postprocessor_args_float32() {
        let args: PostProcessorArgs = AudioFormat::Float32.into();
        assert_eq!(
            args.ffmpeg,
            ["-ar", "16000", "-ac", "1", "-c:a", "pcm_f32le"]
        );
    }

    #[test]
    fn dl_options_from_audio_format() {
        let opts: DownloadParams = AudioFormat::Pcm16.into();

        match opts {
            DownloadParams {
                format: Some(format),
                paths: Some(_),
                outtmpl: Some(_),
                postprocessors: Some(_),
                postprocessor_args: Some(_),
                writeinfojson: Some(true),
                restrictfilenames: Some(true),
                getcomments: None,
                quiet: None,
                no_warnings: None,
                keepvideo: Some(true),
            } if format == "ba" => {}
            _ => panic!(),
        }
    }

    #[test]
    fn audio_format_default() {
        assert!(matches!(AudioFormat::default(), AudioFormat::Pcm16));
    }
}
