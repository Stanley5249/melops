//! Detokenizer for converting token IDs to text segments with timestamps.

use tokenizers::Tokenizer;

/// Detokenizer for TDT models.
///
/// Wraps a HuggingFace tokenizer and caches the blank token ID for efficient
/// access during inference.
pub struct TdtTokenizer {
    /// HuggingFace tokenizer (boxed to reduce struct size).
    pub tokenizer: Box<Tokenizer>,
    /// Blank token ID (equal to vocab size).
    pub blank_id: usize,
}

impl TdtTokenizer {
    /// Create a new detokenizer from a HuggingFace tokenizer.
    ///
    /// Calculates and caches the blank token ID (vocab_size) for efficient
    /// access during decoding.
    pub fn new(tokenizer: Tokenizer) -> Self {
        let blank_id = tokenizer.get_vocab_size(true);
        Self {
            tokenizer: Box::new(tokenizer),
            blank_id,
        }
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.blank_id
    }
}
