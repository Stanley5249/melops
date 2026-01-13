//! AsrModel trait implementation for TdtModel.

use crate::error::{Error, Result};
use crate::models::tdt::core::TdtModel;
use crate::models::tdt::detokenizer::TokenDuration;
use crate::traits::AsrModel;
use crate::types::Segment;

impl AsrModel for TdtModel {
    /// Model output is individual token-duration items.
    ///
    /// Each `TokenDuration` represents a single decoded token with its frame
    /// position and duration. The `forward` method returns a vector of these items.
    type Output = TokenDuration;

    fn frame_to_secs(&self, frame: usize) -> f32 {
        self.mel.frame_to_secs(frame, Self::SUBSAMPLING_FACTOR)
    }

    fn secs_to_frame(&self, secs: f32) -> usize {
        self.mel.secs_to_frame(secs, Self::SUBSAMPLING_FACTOR)
    }
    /// Run TDT inference on audio, returning token-duration sequence.
    fn forward(&mut self, audio: &[f32]) -> Result<Vec<Self::Output>> {
        let features = self.mel.apply(audio);
        let encoder_output = self.encode(features)?;
        self.greedy_decode(encoder_output)
    }

    /// Convert token-duration sequence to text segments with timestamps.
    fn to_segments(&self, token_durations: &[TokenDuration]) -> Result<Vec<Segment>> {
        let mut stream = self.detokenizer.tokenizer.decode_stream(true);

        let output_to_segment = |td: &TokenDuration| match stream.step(td.token_id as u32) {
            Ok(Some(text)) => Some(Ok(Segment {
                text,
                start: self.frame_to_secs(td.frame_index),
                end: self.frame_to_secs(td.frame_index + td.duration),
            })),
            Ok(None) => None,
            Err(e) => Some(Err(Error::Tokenizers(e))),
        };

        token_durations
            .iter()
            .filter_map(output_to_segment)
            .collect()
    }

    /// Offset frame indices in token-duration sequence for chunked processing.
    fn offset_outputs(outputs: &mut [Self::Output], frame_offset: usize) {
        for td in outputs.iter_mut() {
            td.frame_index += frame_offset;
        }
    }

    /// Merge token-duration sequences from multiple chunks.
    fn merge_outputs(chunks: impl IntoIterator<Item = Vec<Self::Output>>) -> Vec<Self::Output> {
        super::merge::merge_outputs(chunks)
    }
}
