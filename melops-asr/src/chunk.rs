//! Audio chunking utilities for processing long audio files.

use crate::error::{ConfigError, Result};
use std::ops::Range;

/// Default chunk duration in seconds (1 minute)
pub const DEFAULT_CHUNK_DURATION: f32 = 60.0;

/// Default chunk overlap in seconds
pub const DEFAULT_CHUNK_OVERLAP: f32 = 1.0;

/// Configuration for audio chunking.
#[derive(Clone, Copy, Debug)]
pub struct ChunkConfig {
    /// Chunk duration in seconds for long audio
    duration: f32,

    /// Chunk overlap in seconds
    overlap: f32,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        // Safe: constants are valid
        Self::new(DEFAULT_CHUNK_DURATION, DEFAULT_CHUNK_OVERLAP).unwrap()
    }
}

impl ChunkConfig {
    /// Create a new chunk configuration.
    ///
    /// Returns error if overlap >= duration.
    pub fn new(duration: f32, overlap: f32) -> Result<Self> {
        if overlap >= duration {
            return Err(ConfigError::InvalidChunkOverlap { overlap, duration }.into());
        }
        Ok(Self { duration, overlap })
    }

    /// Get chunk duration in seconds.
    pub fn duration(&self) -> f32 {
        self.duration
    }

    /// Get chunk overlap in seconds.
    pub fn overlap(&self) -> f32 {
        self.overlap
    }

    /// Create an iterator over chunk ranges for audio with given length and sample rate.
    pub fn chunk_audio(&self, len: usize, sample_rate: usize) -> Result<ChunkRangeIter> {
        let chunk_size = (self.duration * sample_rate as f32) as usize;
        let overlap_size = (self.overlap * sample_rate as f32) as usize;

        // Float truncation can cause overlap_size >= chunk_size even when overlap < duration
        if overlap_size >= chunk_size {
            return Err(ConfigError::InvalidChunkOverlap {
                overlap: self.overlap,
                duration: self.duration,
            }
            .into());
        }

        let step_size = chunk_size - overlap_size;

        Ok(ChunkRangeIter {
            len,
            chunk_size,
            step_size,
            position: 0,
        })
    }
}

/// Iterator over chunk ranges.
pub struct ChunkRangeIter {
    len: usize,
    chunk_size: usize,
    step_size: usize,
    position: usize,
}

impl ChunkRangeIter {
    /// Get the total number of chunks this iterator will produce.
    pub fn len(&self) -> usize {
        if self.len == 0 || self.step_size == 0 {
            return 0;
        }

        // Calculate total chunks: ceil(len / step_size)
        (self.len + self.step_size - 1) / self.step_size
    }

    /// Check if the iterator is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Iterator for ChunkRangeIter {
    type Item = Range<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.len {
            return None;
        }

        let start = self.position;
        let end = (start + self.chunk_size).min(self.len);

        self.position += self.step_size;

        Some(start..end)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.step_size == 0 {
            return (0, None);
        }

        let remaining = self.len.saturating_sub(self.position);
        if remaining == 0 {
            return (0, Some(0));
        }

        // Calculate chunks remaining: ceil(remaining / step_size)
        let count = (remaining + self.step_size - 1) / self.step_size;
        (count, Some(count))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: usize = 16000;

    // Configuration validation
    #[test]
    fn rejects_invalid_overlap() {
        assert!(ChunkConfig::new(60.0, 60.0).is_err());
        assert!(ChunkConfig::new(60.0, 61.0).is_err());
    }

    #[test]
    fn rejects_float_truncation_to_zero_step() {
        let config = ChunkConfig::new(0.0001, 0.00009).unwrap();
        assert!(config.chunk_audio(10 * SAMPLE_RATE, SAMPLE_RATE).is_err());
    }

    // Empty audio
    #[test]
    fn empty_audio() {
        let config = ChunkConfig::new(60.0, 1.0).unwrap();
        let iter = config.chunk_audio(0, SAMPLE_RATE).unwrap();

        assert!(iter.is_empty());
        assert_eq!(iter.len(), 0);
        assert_eq!(iter.count(), 0);
    }

    // Single chunk
    #[test]
    fn audio_shorter_than_chunk() {
        let config = ChunkConfig::new(60.0, 1.0).unwrap();
        let chunks: Vec<_> = config
            .chunk_audio(30 * SAMPLE_RATE, SAMPLE_RATE)
            .unwrap()
            .collect();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], 0..30 * SAMPLE_RATE);
    }

    // Multiple chunks with overlap
    #[test]
    fn multiple_chunks() {
        let config = ChunkConfig::new(60.0, 1.0).unwrap();
        let chunks: Vec<_> = config
            .chunk_audio(150 * SAMPLE_RATE, SAMPLE_RATE)
            .unwrap()
            .collect();

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], 0..60 * SAMPLE_RATE);
        assert_eq!(chunks[1], 59 * SAMPLE_RATE..119 * SAMPLE_RATE);
        assert_eq!(chunks[2], 118 * SAMPLE_RATE..150 * SAMPLE_RATE);
    }

    // Zero overlap (non-overlapping chunks)
    #[test]
    fn zero_overlap() {
        let config = ChunkConfig::new(60.0, 0.0).unwrap();
        let chunks: Vec<_> = config
            .chunk_audio(120 * SAMPLE_RATE, SAMPLE_RATE)
            .unwrap()
            .collect();

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], 0..60 * SAMPLE_RATE);
        assert_eq!(chunks[1], 60 * SAMPLE_RATE..120 * SAMPLE_RATE);
    }

    // Maximum overlap (minimum step size)
    #[test]
    fn maximum_overlap() {
        let config = ChunkConfig::new(1.0, 0.999).unwrap();
        let iter = config.chunk_audio(10 * SAMPLE_RATE, SAMPLE_RATE).unwrap();

        assert!(iter.step_size > 0);
        assert!(iter.step_size < SAMPLE_RATE / 100);
    }

    // Iterator contract
    #[test]
    fn size_hint_and_len_accurate() {
        let config = ChunkConfig::new(60.0, 1.0).unwrap();
        let iter = config.chunk_audio(150 * SAMPLE_RATE, SAMPLE_RATE).unwrap();

        let total_len = iter.len();
        let (lower, upper) = iter.size_hint();
        let actual_count = iter.count();

        assert_eq!(total_len, actual_count);
        assert_eq!(lower, actual_count);
        assert_eq!(upper, Some(actual_count));
    }

    #[test]
    fn size_hint_updates_correctly() {
        let config = ChunkConfig::new(60.0, 1.0).unwrap();
        let mut iter = config.chunk_audio(150 * SAMPLE_RATE, SAMPLE_RATE).unwrap();

        assert_eq!(iter.size_hint(), (3, Some(3)));
        iter.next();
        assert_eq!(iter.size_hint(), (2, Some(2)));
        iter.next();
        assert_eq!(iter.size_hint(), (1, Some(1)));
        iter.next();
        assert_eq!(iter.size_hint(), (0, Some(0)));
    }
}
