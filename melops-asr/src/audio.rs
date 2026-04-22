//! Audio preprocessing utilities.

use ndarray::prelude::*;
use std::f32::consts::PI;

/// Mel-spectrogram feature extractor.
///
/// Converts raw audio into mel-spectrogram features for ASR inference.
#[derive(Clone, Debug)]
pub struct MelSpectrogram {
    pub n_mels: usize,
    pub hop_length: usize,
    pub n_fft: usize,
    pub preemphasis: f32,
    pub sample_rate: usize,
    pub win_length: usize,
}

impl MelSpectrogram {
    /// Extract mel-spectrogram features from a 16 kHz mono f32 slice.
    ///
    /// # Arguments
    ///
    /// * `audio` - 16kHz mono audio samples (f32 slice)
    ///
    /// # Returns
    ///
    /// 2D array of mel-spectrogram features (time_steps, n_mels)
    pub fn apply(&self, audio: &[f32]) -> Array2<f32> {
        mel_spectrogram(audio, self)
    }

    /// Convert seconds to audio sample count.
    pub fn secs_to_samples(&self, secs: f32) -> usize {
        (secs * self.sample_rate as f32) as usize
    }

    /// Convert audio sample count to seconds.
    pub fn samples_to_secs(&self, samples: usize) -> f32 {
        samples as f32 / self.sample_rate as f32
    }

    /// Convert audio sample index to mel-spectrogram frame index.
    pub fn samples_to_mel_frames(&self, samples: usize) -> usize {
        samples / self.hop_length
    }

    /// Convert mel-spectrogram frame index to audio sample index.
    ///
    /// Mel frames are produced at `hop_length` intervals.
    pub fn mel_frames_to_samples(&self, mel_frames: usize) -> usize {
        mel_frames * self.hop_length
    }
}

/// Apply preemphasis filter to audio signal.
///
/// Enhances high frequencies by applying: `y[i] = x[i] - coef * x[i-1]`
fn apply_preemphasis(audio: &[f32], coef: f32) -> Vec<f32> {
    let mut result = Vec::with_capacity(audio.len());
    result.push(audio[0]);

    for i in 1..audio.len() {
        result.push(audio[i] - coef * audio[i - 1]);
    }

    result
}

/// Create Hann window for STFT.
fn hann_window(window_length: usize) -> Vec<f32> {
    (0..window_length)
        .map(|i| 0.5 - 0.5 * ((2.0 * PI * i as f32) / (window_length as f32 - 1.0)).cos())
        .collect()
}

/// Compute Short-Time Fourier Transform (STFT) power spectrogram.
///
/// Uses RustFFT for O(n log n) performance with numerically correct results.
fn stft(audio: &[f32], n_fft: usize, hop_length: usize, win_length: usize) -> Array2<f32> {
    use rustfft::{FftPlanner, num_complex::Complex};

    let window = hann_window(win_length);
    let num_frames = (audio.len() - win_length) / hop_length + 1;
    let freq_bins = n_fft / 2 + 1;
    let mut spectrogram = Array2::<f32>::zeros((freq_bins, num_frames));

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_length;

        let mut frame: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); n_fft];
        for i in 0..win_length.min(audio.len() - start) {
            frame[i] = Complex::new(audio[start + i] * window[i], 0.0);
        }

        fft.process(&mut frame);

        for k in 0..freq_bins {
            let magnitude = frame[k].norm();
            spectrogram[[k, frame_idx]] = magnitude * magnitude;
        }
    }

    spectrogram
}

/// Convert frequency in Hz to mel scale.
fn hz_to_mel(freq: f32) -> f32 {
    2595.0 * (1.0 + freq / 700.0).log10()
}

/// Convert mel scale to frequency in Hz.
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Create mel filterbank for converting STFT to mel spectrogram.
fn create_mel_filterbank(n_fft: usize, n_mels: usize, sample_rate: usize) -> Array2<f32> {
    let freq_bins = n_fft / 2 + 1;
    let mut filterbank = Array2::<f32>::zeros((n_mels, freq_bins));

    let min_mel = hz_to_mel(0.0);
    let max_mel = hz_to_mel(sample_rate as f32 / 2.0);

    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_to_hz(min_mel + (max_mel - min_mel) * i as f32 / (n_mels + 1) as f32))
        .collect();

    let freq_bin_width = sample_rate as f32 / n_fft as f32;

    for mel_idx in 0..n_mels {
        let left = mel_points[mel_idx];
        let center = mel_points[mel_idx + 1];
        let right = mel_points[mel_idx + 2];

        for freq_idx in 0..freq_bins {
            let freq = freq_idx as f32 * freq_bin_width;

            if freq >= left && freq <= center {
                filterbank[[mel_idx, freq_idx]] = (freq - left) / (center - left);
            } else if freq > center && freq <= right {
                filterbank[[mel_idx, freq_idx]] = (right - freq) / (right - center);
            }
        }
    }

    filterbank
}

/// Extract mel-spectrogram features from audio samples.
///
/// Performs complete preprocessing pipeline:
/// 1. Applies preemphasis filter
/// 2. Computes STFT power spectrogram
/// 3. Applies mel filterbank
/// 4. Log compression
/// 5. Mean-variance normalization per feature
///
/// Internal function - prefer using `MelSpectrogram::apply()`.
///
/// # Arguments
///
/// * `audio` - 16kHz mono audio samples (f32 slice)
/// * `config` - Mel-spectrogram configuration
///
/// # Returns
///
/// 2D array of mel-spectrogram features (time_steps, n_mels)
fn mel_spectrogram(audio: &[f32], config: &MelSpectrogram) -> Array2<f32> {
    let audio = apply_preemphasis(audio, config.preemphasis);

    let spectrogram = stft(&audio, config.n_fft, config.hop_length, config.win_length);

    let mel_filterbank = create_mel_filterbank(config.n_fft, config.n_mels, config.sample_rate);
    let mel_spectrogram = mel_filterbank.dot(&spectrogram);
    let mel_spectrogram = mel_spectrogram.mapv(|x| (x.max(1e-10)).ln());

    let mut mel_spectrogram = mel_spectrogram.t().to_owned();

    // Normalize each feature dimension to mean=0, std=1
    let num_frames = mel_spectrogram.shape()[0];
    let num_features = mel_spectrogram.shape()[1];

    for feat_idx in 0..num_features {
        let mut column = mel_spectrogram.column_mut(feat_idx);
        let mean: f32 = column.iter().sum::<f32>() / num_frames as f32;
        let variance: f32 =
            column.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / num_frames as f32;
        let std = variance.sqrt().max(1e-10);

        for val in column.iter_mut() {
            *val = (*val - mean) / std;
        }
    }

    mel_spectrogram
}
