//! Core traits for ASR pipeline components.

use crate::error::Result;
use crate::merge::merge_chunks;
use crate::types::{Segment, TokenDuration};
use futures::stream::TryStreamExt;
use melops_audio::segment::{SegmentConfig, SegmentStream};
use tracing::instrument;

/// ASR model that performs inference on preprocessed features.
///
/// This trait abstracts over different model architectures (TDT, CDC, EOU, Whisper)
/// while providing a uniform interface for the pipeline.
#[allow(async_fn_in_trait)]
pub trait AsrModel {
    /// Get the sample rate.
    fn sample_rate(&self) -> usize;

    /// Convert seconds to audio samples index.
    fn secs_to_samples(&self, secs: f32) -> usize;

    /// Convert audio samples index to seconds.
    fn samples_to_secs(&self, samples: usize) -> f32;

    /// Convert audio samples index to encoder frames index.
    fn samples_to_frames(&self, samples: usize) -> usize;

    /// Convert encoder frames index to audio samples index.
    fn frames_to_samples(&self, frames: usize) -> usize;

    /// Convert seconds to encoder output frame index.
    fn secs_to_frame(&self, secs: f32) -> usize {
        let samples = self.secs_to_samples(secs);
        self.samples_to_frames(samples)
    }

    /// Convert encoder output frame index to seconds.
    fn frame_to_secs(&self, frame: usize) -> f32 {
        let samples = self.frames_to_samples(frame);
        self.samples_to_secs(samples)
    }

    /// Run inference on the given audio, returning a sequence of tokens with timing.
    async fn forward(&self, audio: &[f32]) -> Result<Vec<TokenDuration>>;

    /// Run inference on a chunk with absolute sample-offset adjustment.
    async fn forward_chunk(&self, samples: &[f32], offset: usize) -> Result<Vec<TokenDuration>> {
        let frames = self.samples_to_frames(offset);

        let mut output = self.forward(samples).await?;
        for token in &mut output {
            token.frame_index += frames;
        }

        Ok(output)
    }

    /// Convert a sequence of model outputs to text segments with timestamps.
    fn to_segments(&self, output: &[TokenDuration]) -> Result<Vec<Segment>>;

    /// Transcribe audio from any chunk source, returning merged segments.
    #[instrument(skip_all)]
    async fn transcribe<T: SegmentStream>(
        &self,
        mut stream: T,
        config: SegmentConfig,
    ) -> Result<Vec<Segment>> {
        let mut stream = std::pin::pin!(stream.stream(config));

        let mut chunks: Vec<Vec<TokenDuration>> = Vec::new();
        let mut offset = 0;
        let mut i = 1usize;

        while let Some(segment) = stream.try_next().await? {
            tracing::info!(
                segment = i,
                range = %format!(
                    "{:.1}-{:.1}s",
                    self.samples_to_secs(offset),
                    self.samples_to_secs(offset + segment.len())
                ),
            );

            let output = self.forward_chunk(&segment, offset).await?;

            offset += config.step_size();
            i += 1;

            chunks.push(output);
        }

        let merged = merge_chunks(chunks);

        self.to_segments(&merged)
    }
}
