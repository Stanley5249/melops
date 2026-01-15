//! Merge overlapping token-duration chunks.

use crate::types::TokenDuration;

/// Merge multiple token-duration chunks with frame-based overlap detection.
pub fn merge_chunks<I>(chunks: I) -> Vec<TokenDuration>
where
    I: IntoIterator<Item = Vec<TokenDuration>>,
{
    chunks.into_iter().fold(Vec::new(), merge_two)
}

fn merge_two(mut chunk1: Vec<TokenDuration>, chunk2: Vec<TokenDuration>) -> Vec<TokenDuration> {
    if chunk2.is_empty() {
        return chunk1;
    }

    let last_token = match chunk1.last() {
        Some(td) => td,
        None => return chunk2,
    };

    let chunk1_end_frame = last_token.frame_index + last_token.duration;

    if let Some(i) = chunk2
        .iter()
        .position(|td| td.frame_index >= chunk1_end_frame)
    {
        chunk1.extend_from_slice(&chunk2[i..]);
    }

    chunk1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_empty_chunks() {
        let chunk1 = vec![];
        let chunk2 = vec![TokenDuration {
            token_id: 1,
            frame_index: 0,
            duration: 10,
        }];

        let result = merge_chunks([chunk1, chunk2]);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].token_id, 1);
    }

    #[test]
    fn merges_with_frame_overlap() {
        let chunk1 = vec![TokenDuration::new(1, 0, 10), TokenDuration::new(2, 10, 10)];

        let chunk2 = vec![
            TokenDuration::new(2, 15, 5),
            TokenDuration::new(3, 20, 5),
            TokenDuration::new(4, 25, 5),
        ];

        let result = merge_chunks([chunk1, chunk2]);

        assert_eq!(result.len(), 4);
        assert_eq!(result[0].frame_index, 0);
        assert_eq!(result[1].frame_index, 10);
        assert_eq!(result[2].frame_index, 20);
        assert_eq!(result[3].frame_index, 25);
    }

    #[test]
    fn merges_at_boundary() {
        let chunk1 = vec![TokenDuration::new(1, 0, 10)];

        let chunk2 = vec![TokenDuration::new(1, 10, 10), TokenDuration::new(2, 20, 5)];

        let result = merge_chunks([chunk1, chunk2]);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].frame_index, 0);
        assert_eq!(result[1].frame_index, 10);
        assert_eq!(result[2].frame_index, 20);
    }
}
