//! Audio chunking utilities for processing long audio files.

use std::ops::Range;

/// Default chunk duration in seconds (1 minute)
pub const DEFAULT_CHUNK_DURATION: f32 = 60.0;

/// Default chunk overlap in seconds
pub const DEFAULT_CHUNK_OVERLAP: f32 = 1.0;

/// Configuration for audio chunking.
#[derive(Clone, Copy, Debug)]
pub struct ChunkConfig {
    /// Chunk duration in seconds for long audio
    pub duration: f32,

    /// Chunk overlap in seconds
    pub overlap: f32,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            duration: DEFAULT_CHUNK_DURATION,
            overlap: DEFAULT_CHUNK_OVERLAP,
        }
    }
}

impl ChunkConfig {
    /// Create a new chunk configuration.
    pub fn new(duration: f32, overlap: f32) -> Self {
        Self { duration, overlap }
    }

    /// Create an iterator over chunk ranges for audio with given length and sample rate.
    pub fn chunk_audio(&self, len: usize, sample_rate: usize) -> ChunkRangeIter {
        let chunk_size = (self.duration * sample_rate as f32) as usize;
        let overlap_size = (self.overlap * sample_rate as f32) as usize;
        let step_size = chunk_size.saturating_sub(overlap_size);

        ChunkRangeIter {
            len,
            chunk_size,
            step_size,
            position: 0,
        }
    }
}

/// Iterator over chunk ranges.
pub struct ChunkRangeIter {
    len: usize,
    chunk_size: usize,
    step_size: usize,
    position: usize,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: usize = 16000;

    #[test]
    fn single_chunk_when_audio_shorter_than_duration() {
        let config = ChunkConfig::new(60.0, 1.0);
        let len = 30 * SAMPLE_RATE; // 30 seconds

        let mut iter = config.chunk_audio(len, SAMPLE_RATE);

        assert_eq!(iter.next(), Some(0..len));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn multiple_chunks_with_overlap() {
        let config = ChunkConfig::new(60.0, 1.0);
        let len = 150 * SAMPLE_RATE; // 150 seconds

        let chunks: Vec<_> = config.chunk_audio(len, SAMPLE_RATE).collect();

        // Step is 59 seconds (60 - 1)
        // Chunks at: 0, 59, 118
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], 0..60 * SAMPLE_RATE);
        assert_eq!(chunks[1], 59 * SAMPLE_RATE..119 * SAMPLE_RATE);
        assert_eq!(chunks[2], 118 * SAMPLE_RATE..150 * SAMPLE_RATE);
    }

    #[test]
    fn exact_boundary() {
        let config = ChunkConfig::new(60.0, 1.0);
        let len = 118 * SAMPLE_RATE; // Exactly at step boundary

        let chunks: Vec<_> = config.chunk_audio(len, SAMPLE_RATE).collect();

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], 0..60 * SAMPLE_RATE);
        assert_eq!(chunks[1], 59 * SAMPLE_RATE..118 * SAMPLE_RATE);
    }

    #[test]
    fn zero_overlap() {
        let config = ChunkConfig::new(60.0, 0.0);
        let len = 120 * SAMPLE_RATE;

        let chunks: Vec<_> = config.chunk_audio(len, SAMPLE_RATE).collect();

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], 0..60 * SAMPLE_RATE);
        assert_eq!(chunks[1], 60 * SAMPLE_RATE..120 * SAMPLE_RATE);
    }
}
