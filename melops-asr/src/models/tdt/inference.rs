//! ONNX inference for TDT encoder and decoder-joint.

use crate::error::{ModelError, Result};
use crate::models::tdt::core::TdtModel;
use crate::types::TokenDuration;
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use ort::inputs;
use ort::session::RunOptions;
use ort::value::Tensor;
use tracing::instrument;

/// Encoder outputs and sequence lengths.
pub(super) struct EncoderResponse {
    /// Encoder outputs (shape: (1, encoder_dim, time_steps)).
    encoder_frames: Tensor<f32>,
    /// Encoded sequence lengths.
    encoded_lengths: Tensor<i64>,
}

/// Decoder-joint outputs.
struct DecoderResponse {
    /// Logits (shape: (1, time, 1, vocab_size + num_durations)).
    logits: Tensor<f32>,
    /// LSTM state 1.
    states_1: Tensor<f32>,
    /// LSTM state 2.
    states_2: Tensor<f32>,
}

/// Decoder state.
struct DecoderState {
    /// Current target token.
    target: Tensor<i32>,
    /// Target length (always 1).
    target_length: Tensor<i32>,
    /// LSTM state 1.
    states_1: Tensor<f32>,
    /// LSTM state 2.
    states_2: Tensor<f32>,
}

impl TdtModel {
    #[instrument(skip_all)]
    pub(super) async fn encode(&self, audio_signal: Array2<f32>) -> Result<EncoderResponse> {
        let time_steps = audio_signal.dim().0;
        let audio_lengths = Tensor::from_array(([1_usize], vec![time_steps as i64]))?;

        // (time, features) → (1, features, time)
        let audio_signal = audio_signal.reversed_axes().insert_axis(Axis(0));
        let audio_signal = Tensor::from_array(audio_signal)?;

        let inputs = inputs![
            "audio_signal" => audio_signal,
            "length" => audio_lengths,
        ];

        let options = RunOptions::new()?.with_tag("encoder")?;

        // Mutex guard must be held across await due to lifetime requirements
        let mut encoder_guard = self.encoder.lock().await;

        let mut outputs = encoder_guard.run_async(inputs, &options)?.await?;

        let encoder_frames = outputs
            .remove("outputs")
            .ok_or_else(|| ModelError::missing_output("outputs"))?
            .downcast()?;

        let encoded_lengths = outputs
            .remove("encoded_lengths")
            .ok_or_else(|| ModelError::missing_output("encoded_lengths"))?
            .downcast()?;

        Ok(EncoderResponse {
            encoder_frames,
            encoded_lengths,
        })
    }

    #[instrument(skip_all)]
    pub(super) async fn greedy_decode(
        &self,
        response: EncoderResponse,
    ) -> Result<Vec<TokenDuration>> {
        let decoder_options = RunOptions::new()?.with_tag("decoder_joint")?;

        let mut state = DecoderState {
            target: Tensor::from_array(([1, 1], vec![self.tokenizer.blank_id as i32]))?,
            target_length: Tensor::from_array(([1], vec![1_i32]))?,
            states_1: Tensor::from_array(Array3::<f32>::zeros((2, 1, 640)))?,
            states_2: Tensor::from_array(Array3::<f32>::zeros((2, 1, 640)))?,
        };

        let frames = response
            .encoder_frames
            .extract_array()
            .into_dimensionality::<Ix3>()?;

        let encoded_length = (response.encoded_lengths[[0]] as usize).min(frames.dim().2);

        let mut tokens = Vec::new();
        let mut frame_index = 0;

        while frame_index < encoded_length {
            let slice = frame_index..frame_index + 1;

            let frame = Tensor::from_array(frames.slice_axis(Axis(2), slice.into()).to_owned())?;

            frame_index = self
                .label_loop(
                    frame_index,
                    &frame,
                    &mut state,
                    &mut tokens,
                    &decoder_options,
                )
                .await?
                .min(encoded_length);
        }

        Ok(tokens)
    }

    async fn decode(
        &self,
        frame: &Tensor<f32>,
        state: &DecoderState,
        options: &RunOptions,
    ) -> Result<DecoderResponse> {
        let inputs = inputs![
            "encoder_outputs" => frame,
            "targets" => &state.target,
            "target_length" => &state.target_length,
            "input_states_1" => &state.states_1,
            "input_states_2" => &state.states_2
        ];

        let mut decoder_guard = self.decoder_joint.lock().await;

        let mut outputs = decoder_guard.run_async(inputs, options)?.await?;

        let logits = outputs
            .remove("outputs")
            .ok_or_else(|| ModelError::missing_output("outputs"))?
            .downcast()?;

        let states_1 = outputs
            .remove("output_states_1")
            .ok_or_else(|| ModelError::missing_output("output_states_1"))?
            .downcast()?;

        let states_2 = outputs
            .remove("output_states_2")
            .ok_or_else(|| ModelError::missing_output("output_states_2"))?
            .downcast()?;

        Ok(DecoderResponse {
            logits,
            states_1,
            states_2,
        })
    }

    fn parse_logits(&self, logits: &Tensor<f32>) -> Result<(usize, usize)> {
        let logits_view = logits.extract_array();
        let logits_flat = logits_view.flatten();

        let text_logits = logits_flat.slice_axis(Axis(0), (0..self.tokenizer.blank_id + 1).into());
        let token_id = text_logits.argmax()?;

        let duration_logits =
            logits_flat.slice_axis(Axis(0), (self.tokenizer.blank_id + 1..).into());
        let duration_idx = duration_logits.argmax()?;

        let duration = Self::DURATIONS.get(duration_idx).copied().ok_or_else(|| {
            ModelError::DurationIndexOutOfBounds {
                index: duration_idx,
                max: Self::DURATIONS.len() - 1,
            }
        })?;

        Ok((token_id, duration))
    }

    async fn label_loop(
        &self,
        frame_index: usize,
        frame: &Tensor<f32>,
        state: &mut DecoderState,
        tokens: &mut Vec<TokenDuration>,
        decoder_options: &RunOptions,
    ) -> Result<usize> {
        for _ in 0..Self::MAX_TOKENS_PER_FRAME {
            let response = self.decode(frame, state, decoder_options).await?;

            let (token_id, duration) = self.parse_logits(&response.logits)?;

            if token_id != self.tokenizer.blank_id {
                // With fresh allocations, we can use direct assignment
                state.states_1 = response.states_1;
                state.states_2 = response.states_2;

                tokens.push(TokenDuration::new(token_id, frame_index, duration));

                state.target[[0, 0]] = token_id as i32;
            }

            if duration != 0 {
                return Ok(frame_index + duration);
            }
        }

        Ok(frame_index + 1)
    }
}
