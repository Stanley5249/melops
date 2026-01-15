//! Detokenizer for converting token IDs to text segments with timestamps.

use tokenizers::Tokenizer;

/// Detokenizer for TDT models.
pub struct TdtDetokenizer {
    pub tokenizer: Tokenizer,
}

impl TdtDetokenizer {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self { tokenizer }
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }
}
