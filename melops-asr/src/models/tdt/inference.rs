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
pub(super) struct TdtEncoderOutputs {
    /// Encoder outputs (shape: (1, encoder_dim, time_steps)).
    encoder_outputs: Tensor<f32>,
    /// Encoded sequence lengths.
    encoded_lengths: Tensor<i64>,
}

/// Decoder-joint outputs.
struct TdtDecoderJointOutputs {
    /// Logits (shape: (1, time, 1, vocab_size + num_durations)).
    outputs: Tensor<f32>,
    /// LSTM state 1.
    output_states_1: Tensor<f32>,
    /// LSTM state 2.
    output_states_2: Tensor<f32>,
}

/// Decoded token and duration.
struct DecodedOutput {
    /// Token ID.
    token_id: usize,
    /// Duration (frames to skip).
    duration: usize,
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
    pub(super) async fn encode(&self, audio_signal: Array2<f32>) -> Result<TdtEncoderOutputs> {
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

        let encoder_outputs = outputs
            .remove("outputs")
            .ok_or_else(|| ModelError::missing_output("outputs"))?
            .downcast()?;

        let encoded_lengths = outputs
            .remove("encoded_lengths")
            .ok_or_else(|| ModelError::missing_output("encoded_lengths"))?
            .downcast()?;

        Ok(TdtEncoderOutputs {
            encoder_outputs,
            encoded_lengths,
        })
    }

    #[instrument(skip_all)]
    pub(super) async fn greedy_decode(
        &self,
        values: TdtEncoderOutputs,
    ) -> Result<Vec<TokenDuration>> {
        let decoder_options = RunOptions::new()?.with_tag("decoder_joint")?;

        let mut state = DecoderState {
            target: Tensor::from_array(([1, 1], vec![self.tokenizer.blank_id as i32]))?,
            target_length: Tensor::from_array(([1], vec![1_i32]))?,
            states_1: Tensor::from_array(Array3::<f32>::zeros((2, 1, 640)))?,
            states_2: Tensor::from_array(Array3::<f32>::zeros((2, 1, 640)))?,
        };

        let frames = values
            .encoder_outputs
            .extract_array()
            .into_dimensionality::<Ix3>()?;

        let encoded_length =
            (values.encoded_lengths.extract_array()[0] as usize).min(frames.dim().2);

        let mut tokens = Vec::new();
        let mut frame_index = 0;

        while frame_index < encoded_length {
            let frame = frames
                .slice_axis(Axis(2), (frame_index..frame_index + 1).into())
                .into_owned();
            let frame = Tensor::from_array(frame)?;

            frame_index = self
                .label_loop(
                    frame_index,
                    encoded_length,
                    &frame,
                    &mut state,
                    &decoder_options,
                    &mut tokens,
                )
                .await?;
        }

        Ok(tokens)
    }

    async fn decode(
        &self,
        frame: &Tensor<f32>,
        state: &DecoderState,
        options: &RunOptions,
    ) -> Result<TdtDecoderJointOutputs> {
        let inputs = inputs![
            "encoder_outputs" => frame,
            "targets" => &state.target,
            "target_length" => &state.target_length,
            "input_states_1" => &state.states_1,
            "input_states_2" => &state.states_2
        ];

        let mut decoder_guard = self.decoder_joint.lock().await;

        let mut session_outputs = decoder_guard.run_async(inputs, options)?.await?;

        let outputs = session_outputs
            .remove("outputs")
            .ok_or_else(|| ModelError::missing_output("outputs"))?
            .downcast()?;

        let output_states_1 = session_outputs
            .remove("output_states_1")
            .ok_or_else(|| ModelError::missing_output("output_states_1"))?
            .downcast()?;

        let output_states_2 = session_outputs
            .remove("output_states_2")
            .ok_or_else(|| ModelError::missing_output("output_states_2"))?
            .downcast()?;

        Ok(TdtDecoderJointOutputs {
            outputs,
            output_states_1,
            output_states_2,
        })
    }

    fn parse_logits(&self, logits: &Tensor<f32>) -> Result<DecodedOutput> {
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

        Ok(DecodedOutput { token_id, duration })
    }

    async fn label_loop(
        &self,
        mut frame_index: usize,
        encoded_length: usize,
        frame: &Tensor<f32>,
        state: &mut DecoderState,
        decoder_options: &RunOptions,
        tokens: &mut Vec<TokenDuration>,
    ) -> Result<usize> {
        for _ in 0..Self::MAX_TOKENS_PER_FRAME {
            let decoder_outputs = self.decode(frame, state, decoder_options).await?;

            let decoded = self.parse_logits(&decoder_outputs.outputs)?;

            let skip = decoded.duration;

            if decoded.token_id != self.tokenizer.blank_id {
                // With fresh allocations, we can use direct assignment
                state.states_1 = decoder_outputs.output_states_1;
                state.states_2 = decoder_outputs.output_states_2;

                tokens.push(TokenDuration::new(decoded.token_id, frame_index, skip));

                state.target[[0, 0]] = decoded.token_id as i32;
            }

            frame_index = encoded_length.min(frame_index + skip);

            if skip != 0 {
                return Ok(frame_index);
            }
        }

        Ok(frame_index + 1)
    }
}
