//! Audio loading and preprocessing utilities.

use crate::error::{AudioError, Result};
use hound::{SampleFormat, WavReader, WavSpec};
use ndarray::prelude::*;
use std::f32::consts::PI;
use std::path::Path;

/// Expected sample rate for ASR models (16kHz)
pub const SAMPLE_RATE: u32 = 16000;

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
    /// Apply mel-spectrogram extraction to audio samples.
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

/// Load audio from a WAV file.
///
/// Returns audio samples and WAV specification.
///
/// # Errors
///
/// Returns error if file cannot be read or has unsupported format.
pub fn load_audio<P: AsRef<Path>>(path: P) -> Result<(Vec<f32>, WavSpec)> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader.samples::<f32>().collect::<hound::Result<_>>()?,
        SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / i16::MAX as f32))
            .collect::<hound::Result<_>>()?,
    };

    Ok((samples, spec))
}

/// Load audio from a WAV file as mono f32 samples at 16kHz.
///
/// Validates sample rate is 16kHz and converts stereo to mono if needed.
///
/// # Errors
///
/// Returns error if:
/// - File cannot be read
/// - Sample rate is not 16kHz
/// - Channel count is invalid (0 or > 2)
pub fn read_audio_mono(path: impl AsRef<Path>) -> Result<Vec<f32>> {
    let (mut audio, spec) = load_audio(path)?;

    if spec.sample_rate != SAMPLE_RATE {
        return Err(AudioError::InvalidSampleRate {
            expected: SAMPLE_RATE,
            got: spec.sample_rate,
        }
        .into());
    }

    if spec.channels == 0 || spec.channels > 2 {
        return Err(AudioError::InvalidChannels(spec.channels).into());
    }

    if spec.channels == 2 {
        audio = audio
            .chunks(2)
            .map(|chunk| chunk.iter().sum::<f32>() / 2.0)
            .collect();
    }

    Ok(audio)
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

#[cfg(test)]
mod tests {
    use super::*;
    use hound::WavWriter;

    fn create_test_wav(
        path: &Path,
        sample_rate: u32,
        channels: u16,
        samples: &[f32],
    ) -> hound::Result<()> {
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(path, spec)?;
        for &sample in samples {
            writer.write_sample((sample * 32768.0) as i16)?;
        }
        writer.finalize()?;
        Ok(())
    }

    #[test]
    fn reads_mono_16khz() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_mono.wav");

        let test_samples = vec![0.1, 0.2, 0.3];
        create_test_wav(&path, 16000, 1, &test_samples).unwrap();

        let result = read_audio_mono(&path).unwrap();

        for (expected, actual) in test_samples.iter().zip(result.iter()) {
            assert!((expected - actual).abs() < 0.01);
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn converts_stereo_to_mono() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_stereo.wav");

        let test_samples = vec![0.2, 0.4, 0.6, 0.8];
        create_test_wav(&path, 16000, 2, &test_samples).unwrap();

        let result = read_audio_mono(&path).unwrap();

        assert_eq!(result.len(), 2);
        assert!((result[0] - 0.3).abs() < 0.01);
        assert!((result[1] - 0.7).abs() < 0.01);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn rejects_wrong_sample_rate() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_44khz.wav");

        create_test_wav(&path, 44100, 1, &[0.0, 0.1]).unwrap();

        let result = read_audio_mono(&path);

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e, crate::error::Error::Audio(_)));
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn rejects_invalid_channels() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_surround.wav");

        create_test_wav(&path, 16000, 6, &[0.0; 12]).unwrap();

        let result = read_audio_mono(&path);

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e, crate::error::Error::Audio(_)));
        }

        std::fs::remove_file(path).ok();
    }
}
