//! Core traits for ASR pipeline components.

use crate::chunk::ChunkConfig;
use crate::error::Result;
use crate::types::Segment;

/// ASR model that performs inference on preprocessed features.
///
/// This trait abstracts over different model architectures (TDT, CDC, EOU, Whisper)
/// while providing a uniform interface for the pipeline.
pub trait AsrModel {
    /// Output type from model inference.
    ///
    /// Represents a single unit of model output (e.g., a token with timing).
    /// The `forward` method returns `Vec<Self::Output>`, a sequence of these items.
    type Output;

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

    /// Run inference on the given audio, returning a sequence of output items.
    ///
    /// Returns `Vec<Self::Output>` where each item represents a decoded unit
    /// (e.g., token with timing information).
    ///
    /// Note: Takes `&mut self` because ONNX Runtime's Session::run requires it.
    fn forward(&mut self, audio: &[f32]) -> Result<Vec<Self::Output>>;

    /// Convert a sequence of model outputs to text segments with timestamps.
    fn to_segments(&self, output: &[Self::Output]) -> Result<Vec<Segment>>;

    /// Apply frame offset to a sequence of model outputs.
    ///
    /// Used for chunked transcription where frame indices need to be offset
    /// to represent absolute positions in the full audio.
    fn offset_outputs(output: &mut [Self::Output], frames: usize);

    /// Merge output sequences from multiple chunks into a single sequence.
    fn merge_chunks(chunks: impl IntoIterator<Item = Vec<Self::Output>>) -> Vec<Self::Output>;

    /// Transcribe audio samples, returning segments.
    fn transcribe(&mut self, audio: &[f32]) -> Result<Vec<Segment>> {
        let output = self.forward(audio)?;
        self.to_segments(&output)
    }

    /// Transcribe audio with automatic chunking, returning merged segments.
    fn transcribe_chunked(&mut self, audio: &[f32], config: ChunkConfig) -> Result<Vec<Segment>> {
        let outputs = config
            .chunk_audio(audio.len(), self.sample_rate())
            .map(|range| {
                let start = self.samples_to_secs(range.start);
                let end = self.samples_to_secs(range.end);
                tracing::debug!(
                    start=%format!("{start:.2}s"),
                    end=%format!("{end:.2}s"),
                    "transcribe chunk"
                );

                let frames = self.samples_to_frames(range.start);

                let chunk = &audio[range];

                let mut output = self.forward(chunk)?;

                Self::offset_outputs(&mut output, frames);

                Ok(output)
            })
            .collect::<Result<Vec<_>>>()?;

        let merged_output = Self::merge_chunks(outputs);
        self.to_segments(&merged_output)
    }

    /// Transcribe audio from an iterator stream, returning merged segments.
    #[allow(unused_variables)]
    fn transcribe_stream(
        &mut self,
        audio: impl Iterator<Item = f32>,
        config: ChunkConfig,
    ) -> Result<Vec<Segment>> {
        todo!()
    }
}
