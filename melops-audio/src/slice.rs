use crate::segment::{SegmentConfig, SegmentIterator};

impl SegmentIterator for [f32] {
    fn iter(&mut self, config: SegmentConfig) -> impl Iterator<Item = &[f32]> {
        Segmentor::new(self, config)
    }
}

/// Iterator over chunk ranges.
pub struct Segmentor<'a, T> {
    slice: &'a [T],
    cursor: usize,
    config: SegmentConfig,
}

impl<'a, T> Segmentor<'a, T> {
    pub fn new(slice: &'a [T], config: SegmentConfig) -> Self {
        Self {
            slice,
            cursor: 0,
            config,
        }
    }
}

impl<'a, T> Iterator for Segmentor<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.slice.len() {
            return None;
        }

        let i = self.cursor;
        let j = (i + self.config.window_size()).min(self.slice.len());

        self.cursor += self.config.step_size();

        Some(&self.slice[i..j])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<'a, T> ExactSizeIterator for Segmentor<'a, T> {
    fn len(&self) -> usize {
        let r = self.slice.len() - self.cursor;
        r.div_ceil(self.config.step_size())
    }
}
