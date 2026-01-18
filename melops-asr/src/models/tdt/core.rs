//! Core TDT model definition and loading.

use crate::audio::MelSpectrogram;
use crate::models::tdt::tokenizer::TdtTokenizer;
use crate::types::ModelRepo;
use eyre::{Result, WrapErr, eyre};
use ort::session::Session;
use ort::session::builder::SessionBuilder;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::instrument;

/// TDT model for ASR inference.
///
/// Implements the Token-and-Duration Transducer architecture with encoder and
/// joint decoder components. The decoder predicts both tokens and their durations,
/// enabling efficient streaming inference by skipping multiple frames at once.
pub struct TdtModel {
    pub encoder: Arc<Mutex<Session>>,
    pub decoder_joint: Arc<Mutex<Session>>,
    pub tokenizer: TdtTokenizer,
}

impl TdtModel {
    /// TDT model mel-spectrogram extractor (128 mel features).
    pub const MEL_SPECTOGRAM: MelSpectrogram = MelSpectrogram {
        n_mels: 128,
        hop_length: 160,
        n_fft: 512,
        preemphasis: 0.97,
        sample_rate: 16000,
        win_length: 400,
    };

    /// TDT encoder subsampling factor (8x).
    ///
    /// The encoder downsamples the mel-spectrogram by a factor of 8,
    /// producing one encoder frame for every 8 mel frames.
    pub const SUBSAMPLING_FACTOR: usize = 8;

    /// Duration prediction values for TDT decoder.
    ///
    /// The decoder predicts how many encoder frames to skip for each token.
    /// Values represent the number of frames to advance (0-4).
    pub const DURATIONS: [usize; 5] = [0, 1, 2, 3, 4];

    /// Maximum number of tokens to emit per encoder frame.
    ///
    /// Limits consecutive token predictions within a single frame to prevent
    /// infinite loops during greedy decoding.
    pub const MAX_TOKENS_PER_FRAME: usize = 10;

    /// Create a new TDT model instance.
    pub fn new(encoder: Session, decoder_joint: Session, tokenizer: TdtTokenizer) -> Self {
        Self {
            encoder: Arc::new(Mutex::new(encoder)),
            decoder_joint: Arc::new(Mutex::new(decoder_joint)),
            tokenizer,
        }
    }

    /// Load TDT model from a model repository.
    #[instrument(skip_all)]
    pub fn from_repo(repo: &ModelRepo, session_builder: SessionBuilder) -> Result<Self> {
        let encoder_path = repo.resolve_any(&[
            "encoder-model.onnx",
            "encoder.onnx",
            "encoder-model.int8.onnx",
        ])?;

        let decoder_path = repo.resolve_any(&[
            "decoder_joint-model.onnx",
            "decoder_joint.onnx",
            "decoder_joint-model.int8.onnx",
        ])?;

        let tokenizer_path = repo.resolve("tokenizer.json")?;

        let model = TdtModel::new(
            Self::load_encoder(&encoder_path, &session_builder)?,
            Self::load_decoder_joint(&decoder_path, &session_builder)?,
            Self::load_detokenizer(&tokenizer_path)?,
        );

        Ok(model)
    }

    /// Load encoder ONNX session.
    #[instrument(level = "debug", skip_all, fields(file=?file.display()))]
    fn load_encoder(file: &Path, session_builder: &SessionBuilder) -> Result<Session> {
        session_builder
            .clone()
            .commit_from_file(file)
            .wrap_err("failed to load encoder session")
    }

    /// Load decoder ONNX session.
    #[instrument(level = "debug", skip_all, fields(file=?file.display()))]
    fn load_decoder_joint(file: &Path, session_builder: &SessionBuilder) -> Result<Session> {
        session_builder
            .clone()
            .commit_from_file(file)
            .wrap_err("failed to load decoder session")
    }

    /// Load and create detokenizer.
    #[instrument(level = "debug", skip_all, fields(file=?file.display()))]
    fn load_detokenizer(file: &Path) -> Result<TdtTokenizer> {
        let tokenizer = Tokenizer::from_file(file)
            .map_err(|e| eyre!(e))
            .wrap_err("failed to load tokenizer")?;

        Ok(TdtTokenizer::new(tokenizer))
    }
}
