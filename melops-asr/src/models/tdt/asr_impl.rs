//! AsrModel trait implementation for TdtModel.

use crate::error::{Error, Result};
use crate::models::tdt::core::TdtModel;
use crate::traits::AsrModel;
use crate::types::Segment;
use crate::types::TokenDuration;
use tracing::instrument;

impl AsrModel for TdtModel {
    fn sample_rate(&self) -> usize {
        Self::MEL_SPECTOGRAM.sample_rate
    }

    fn secs_to_samples(&self, secs: f32) -> usize {
        Self::MEL_SPECTOGRAM.secs_to_samples(secs)
    }

    fn samples_to_secs(&self, samples: usize) -> f32 {
        Self::MEL_SPECTOGRAM.samples_to_secs(samples)
    }

    fn samples_to_frames(&self, samples: usize) -> usize {
        Self::MEL_SPECTOGRAM.samples_to_mel_frames(samples) / Self::SUBSAMPLING_FACTOR
    }

    fn frames_to_samples(&self, frames: usize) -> usize {
        Self::MEL_SPECTOGRAM.mel_frames_to_samples(frames * Self::SUBSAMPLING_FACTOR)
    }

    /// Run TDT inference on audio, returning token-duration sequence.
    #[instrument(skip_all)]
    async fn forward(&self, audio: &[f32]) -> Result<Vec<TokenDuration>> {
        let features = Self::MEL_SPECTOGRAM.apply(audio);
        let encoder_output = self.encode(features).await?;
        self.greedy_decode(encoder_output).await
    }

    /// Convert token-duration sequence to text segments with timestamps.
    fn to_segments(&self, token_durations: &[TokenDuration]) -> Result<Vec<Segment>> {
        let mut stream = self.tokenizer.tokenizer.decode_stream(true);

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
}
