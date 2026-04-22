//! Error types for melops-asr organized by processing stage.

use ndarray::ShapeError;
use ndarray_stats::errors::{MinMaxError, QuantileError};
use thiserror::Error;

/// ASR pipeline error variants organized by processing stage.
#[derive(Debug, Error)]
pub enum Error {
    /// Audio loading stage error
    #[error(transparent)]
    Audio(#[from] AudioError),

    /// Model inference stage error
    #[error(transparent)]
    Model(#[from] ModelError),

    /// Tokenizer error
    #[error(transparent)]
    Tokenizers(tokenizers::Error),
}

/// Audio loading and validation errors.
#[derive(Debug, Error)]
pub enum AudioError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Model inference errors (ONNX, ndarray operations).
#[derive(Debug, Error)]
pub enum ModelError {
    /// Missing expected session output
    #[error("missing ort output {key:?}")]
    MissingOutput { key: String },

    /// Duration index out of bounds
    #[error("duration index {index} out of bounds (max {max})")]
    DurationIndexOutOfBounds { index: usize, max: usize },

    /// ONNX Runtime error
    #[error(transparent)]
    Ort(#[from] ort::Error),

    /// ndarray shape error
    #[error(transparent)]
    Shape(#[from] ShapeError),

    /// ndarray-stats min/max error
    #[error(transparent)]
    MinMax(#[from] MinMaxError),

    /// ndarray-stats quantile error
    #[error(transparent)]
    Quantile(#[from] QuantileError),
}

impl ModelError {
    pub fn missing_output(key: impl Into<String>) -> Self {
        Self::MissingOutput { key: key.into() }
    }
}

/// Result type alias for melops-asr operations.
pub type Result<T> = std::result::Result<T, Error>;

// std::io::Error → AudioError → Error
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Audio(AudioError::Io(e))
    }
}

// ort::Error → ModelError → Error
impl From<ort::Error> for Error {
    fn from(e: ort::Error) -> Self {
        Error::Model(ModelError::Ort(e))
    }
}

// ShapeError → ModelError → Error
impl From<ShapeError> for Error {
    fn from(e: ShapeError) -> Self {
        Error::Model(ModelError::Shape(e))
    }
}

// MinMaxError → ModelError → Error
impl From<MinMaxError> for Error {
    fn from(e: MinMaxError) -> Self {
        Error::Model(ModelError::MinMax(e))
    }
}

// QuantileError → ModelError → Error
impl From<QuantileError> for Error {
    fn from(e: QuantileError) -> Self {
        Error::Model(ModelError::Quantile(e))
    }
}
