use futures::Stream;
use thiserror::Error;

/// Audio sample rate. FFmpeg always resamples to this rate in the pipeline.
pub const SAMPLE_RATE: usize = 16000;

/// Default segment duration in seconds.
pub const DEFAULT_DURATION: f32 = 30.0;

/// Default overlap in seconds kept between consecutive segments.
pub const DEFAULT_OVERLAP: f32 = 1.0;

#[derive(Clone, Copy, Debug, Error)]
#[error(
    "expected window_size >= step_size > 0, got window_size={window_size}, step_size={step_size}"
)]
pub struct SegmentConfigError {
    pub window_size: usize,
    pub step_size: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct SegmentConfig {
    window_size: usize,
    step_size: usize,
}

impl SegmentConfig {
    pub fn new(window_size: usize, step_size: usize) -> Result<Self, SegmentConfigError> {
        if window_size >= step_size && step_size > 0 {
            Ok(Self {
                window_size,
                step_size,
            })
        } else {
            Err(SegmentConfigError {
                window_size,
                step_size,
            })
        }
    }

    /// Build from human-readable seconds at the pipeline sample rate.
    ///
    /// `duration` is the full window length; `overlap` is the shared tail kept between segments.
    pub fn from_secs(duration: f32, overlap: f32) -> Result<Self, SegmentConfigError> {
        let window_size = (duration * SAMPLE_RATE as f32) as usize;
        let step_size = window_size.saturating_sub((overlap * SAMPLE_RATE as f32) as usize);
        Self::new(window_size, step_size)
    }

    /// window_size >= step_size always holds.
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// step_size > 0 always holds.
    pub fn step_size(&self) -> usize {
        self.step_size
    }

    pub fn overlap_size(&self) -> usize {
        self.window_size - self.step_size
    }
}

pub trait SegmentIterator {
    fn iter(&mut self, config: SegmentConfig) -> impl Iterator<Item = &[f32]>;
}

pub trait SegmentStream {
    fn stream(
        &mut self,
        config: SegmentConfig,
    ) -> impl Stream<Item = std::io::Result<Vec<f32>>> + Send;
}
