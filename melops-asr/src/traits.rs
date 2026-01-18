//! Core traits for ASR pipeline components.

use crate::chunk::ChunkConfig;
use crate::error::Result;
use crate::merge::merge_chunks;
use crate::types::{Segment, TokenDuration};
use futures::stream::{self, StreamExt, TryStreamExt};
use std::ops::Range;
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

    /// Run inference on a chunk of audio with offset, returning model outputs.
    async fn forward_chunk(
        &self,
        audio: &[f32],
        range: Range<usize>,
    ) -> Result<Vec<TokenDuration>> {
        let start = format!("{:.2}s", self.samples_to_secs(range.start));
        let end = format!("{:.2}s", self.samples_to_secs(range.end));
        tracing::info!(%start, %end);

        let frames = self.samples_to_frames(range.start);

        let chunk = &audio[range];

        let mut output = self.forward(chunk).await?;

        // Inline offset_outputs: adjust frame indices by offset
        for token in &mut output {
            token.frame_index += frames;
        }

        Ok(output)
    }

    /// Convert a sequence of model outputs to text segments with timestamps.
    fn to_segments(&self, output: &[TokenDuration]) -> Result<Vec<Segment>>;

    /// Transcribe audio samples, returning segments.
    #[instrument(skip_all)]
    async fn transcribe(&self, audio: &[f32]) -> Result<Vec<Segment>> {
        let output = self.forward(audio).await?;
        self.to_segments(&output)
    }

    /// Transcribe audio with automatic chunking, returning merged segments.
    #[instrument(skip_all)]
    async fn transcribe_chunked(&self, audio: &[f32], config: ChunkConfig) -> Result<Vec<Segment>> {
        let chunk_iter = config.chunk_audio(audio.len(), self.sample_rate())?;
        let total = chunk_iter.len();

        let chunks = chunk_iter
            .zip(1..)
            .map(|(range, i)| {
                tracing::info!(current = i, total);
                range
            })
            .map(|range| self.forward_chunk(audio, range));

        let chunks: Vec<_> = stream::iter(chunks).buffered(2).try_collect().await?;

        let merged_output = merge_chunks(chunks);
        self.to_segments(&merged_output)
    }

    /// Transcribe audio from an iterator stream, returning merged segments.
    #[allow(unused_variables)]
    async fn transcribe_stream(
        &self,
        audio: impl Iterator<Item = f32>,
        config: ChunkConfig,
    ) -> Result<Vec<Segment>> {
        todo!()
    }
}
