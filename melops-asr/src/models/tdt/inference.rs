//! ONNX inference for TDT encoder and decoder-joint.

use crate::error::{ModelError, Result};
use crate::models::tdt::core::TdtModel;
use crate::models::tdt::detokenizer::TokenDuration;
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use ort::io_binding::IoBinding;
use ort::session::RunOptions;
use ort::value::Tensor;
use std::sync::OnceLock;

static DEBUG_ALLOCATOR: OnceLock<()> = OnceLock::new();

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
    pub(super) fn encode(&mut self, audio_signal: Array2<f32>) -> Result<TdtEncoderOutputs> {
        let time_steps = audio_signal.dim().0;
        let audio_lengths = Tensor::from_array(([1_usize], vec![time_steps as i64]))?;

        // (time, features) → (1, features, time)
        let audio_signal = audio_signal.reversed_axes().insert_axis(Axis(0));
        let audio_signal = Tensor::from_array(audio_signal)?;

        let mut binding = self.encoder.create_binding()?;

        binding.bind_input("audio_signal", &audio_signal)?;
        binding.bind_input("length", &audio_lengths)?;

        let allocator = self.encoder.allocator();
        let memory_info = allocator.memory_info();

        DEBUG_ALLOCATOR.get_or_init(|| {
            tracing::debug!(
                allocation_device = ?memory_info.allocation_device(),
                device_id = ?memory_info.device_id(),
                "encoder allocator"
            );
        });

        binding.bind_output_to_device("outputs", memory_info)?;

        let encoded_lengths = Tensor::<i64>::new(allocator, [1_usize])?;
        binding.bind_output("encoded_lengths", encoded_lengths)?;

        let options = RunOptions::new()?.with_tag("encoder")?;

        let mut outputs = self.encoder.run_binding_with_options(&binding, &options)?;

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

    pub(super) fn greedy_decode(
        &mut self,
        values: TdtEncoderOutputs,
    ) -> Result<Vec<TokenDuration>> {
        let blank_id = self.detokenizer.vocab_size();
        let max_symbols_per_step = 10;

        // IoBinding reuses pre-allocated output tensors. Direct assignment
        // would alias state with output_states, causing unintended updates
        // even when blank tokens should skip state changes. Use copy_into
        // for explicit control over when state propagates.
        let mut state = DecoderState {
            target: Tensor::from_array(([1, 1], vec![blank_id as i32]))?,
            target_length: Tensor::from_array(([1], vec![1_i32]))?,
            states_1: Tensor::from_array(Array3::<f32>::zeros((2, 1, 640)))?,
            states_2: Tensor::from_array(Array3::<f32>::zeros((2, 1, 640)))?,
        };

        let mut binding = self.decoder_joint.create_binding()?;

        binding.bind_input("target_length", &state.target_length)?;

        let allocator = self.decoder_joint.allocator();
        let vocab_size = self.detokenizer.vocab_size();
        let num_durations = self.durations.len();

        binding.bind_output(
            "outputs",
            Tensor::<f32>::new(allocator, [1_usize, 1, 1, vocab_size + 1 + num_durations])?,
        )?;
        binding.bind_output(
            "output_states_1",
            Tensor::<f32>::new(allocator, [2_usize, 1, 640])?,
        )?;
        binding.bind_output(
            "output_states_2",
            Tensor::<f32>::new(allocator, [2_usize, 1, 640])?,
        )?;

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

            frame_index = self.label_loop(
                frame_index,
                &frame,
                &mut state,
                &mut binding,
                &mut tokens,
                blank_id,
                max_symbols_per_step,
                encoded_length,
            )?;
        }

        Ok(tokens)
    }

    fn decode(
        &mut self,
        frame: &Tensor<f32>,
        state: &DecoderState,
        binding: &mut IoBinding,
    ) -> Result<TdtDecoderJointOutputs> {
        binding.bind_input("encoder_outputs", frame)?;
        binding.bind_input("targets", &state.target)?;
        binding.bind_input("input_states_1", &state.states_1)?;
        binding.bind_input("input_states_2", &state.states_2)?;

        let options = RunOptions::new()?.with_tag("decoder_joint")?;

        let mut session_outputs = self
            .decoder_joint
            .run_binding_with_options(binding, &options)?;

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

    fn parse_logits(&self, logits: &Tensor<f32>, blank_id: usize) -> Result<DecodedOutput> {
        let logits_view = logits.extract_array();
        let logits_flat = logits_view.flatten();

        let text_logits = logits_flat.slice_axis(Axis(0), (0..blank_id + 1).into());
        let token_id = text_logits.argmax()?;

        let duration_logits = logits_flat.slice_axis(Axis(0), (blank_id + 1..).into());
        let duration_idx = duration_logits.argmax()?;

        let duration = self.durations.get(duration_idx).copied().ok_or_else(|| {
            ModelError::DurationIndexOutOfBounds {
                index: duration_idx,
                max: self.durations.len() - 1,
            }
        })?;

        Ok(DecodedOutput { token_id, duration })
    }

    fn label_loop(
        &mut self,
        mut frame_index: usize,
        frame: &Tensor<f32>,
        state: &mut DecoderState,
        binding: &mut IoBinding,
        tokens: &mut Vec<TokenDuration>,
        blank_id: usize,
        max_symbols_per_step: usize,
        encoded_length: usize,
    ) -> Result<usize> {
        for _ in 0..max_symbols_per_step {
            let decoder_outputs = self.decode(frame, state, binding)?;

            let decoded = self.parse_logits(&decoder_outputs.outputs, blank_id)?;

            let skip = decoded.duration;

            if decoded.token_id != blank_id {
                // With IoBinding, state and output_states alias the same memory.
                // Direct assignment (commented below) only updates Rust references
                // on first iteration. After that, both point to same tensor and
                // ONNX Runtime always writes new states to bound outputs, bypassing
                // this conditional. Use copy_into for explicit control: states only
                // update when we want them to (non-blank tokens).

                // state.states_1 = decoder_outputs.output_states_1;
                // state.states_2 = decoder_outputs.output_states_2;

                decoder_outputs
                    .output_states_1
                    .copy_into(&mut state.states_1)?;
                decoder_outputs
                    .output_states_2
                    .copy_into(&mut state.states_2)?;

                tokens.push(TokenDuration {
                    token_id: decoded.token_id,
                    frame_index,
                    duration: skip,
                });
                state.target[[0, 0]] = decoded.token_id as i32;
            }

            tracing::trace!(frame_index, skip);

            frame_index = encoded_length.min(frame_index + skip);

            if skip != 0 {
                return Ok(frame_index);
            }
        }

        Ok(frame_index + 1)
    }
}
