//! melops-asr: Extensible ASR library with trait-based architecture.
//!
//! This crate provides high-level ASR pipelines while maintaining flexibility
//! for different model architectures (TDT, CDC, EOU, Whisper).
//!
//! # Architecture
//!
//! The library is built around two core traits:
//!
//! - [`traits::AsrModel`]: Runs inference on features to produce model output
//! - [`traits::Detokenizer`]: Converts model output to human-readable transcription
//!
//! Mel-spectrogram extraction is handled by [`audio::MelSpectrogram`].
//!
//! # Quick Start
//!
//! ```ignore
//! use melops_asr::types::AudioBuffer;
//! use melops_asr::pipelines::ParakeetTdt;
//! use ort::session::Session;
//!
//! // Load audio
//! let audio = AudioBuffer::from_file("audio.wav")?;
//!
//! // Create pipeline
//! let builder = Session::builder()?;
//! let pipeline = ParakeetTdt::from_pretrained("model_dir", Some(builder))?;
//!
//! // Transcribe
//! let result = pipeline.transcribe(&audio)?;
//! println!("{}", result.text);
//! ```

pub mod audio;
pub mod chunk;
pub mod error;
pub mod merge;
pub mod models;
pub mod traits;
pub mod types;
