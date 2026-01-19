//! yt-dlp download function and Python API wrapper.
//!
//! Main entry point for downloading media via yt-dlp.
//!
//! ```no_run
//! use melops_dl::{dl::download, asr::AudioFormat};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let (file_path, info) = download("https://youtube.com/watch?v=example", AudioFormat::Pcm16.into())?;
//! println!("Downloaded '{}' to {:?}", info.title, file_path);
//! # Ok(())
//! # }
//! ```

use crate::info::DownloadInfo;
use crate::params::DownloadParams;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use std::path::PathBuf;

/// Downloads media from URL using yt-dlp.
///
/// Returns `(file_path, info)` where `file_path` is the final processed file location.
/// `file_path` is `None` if download failed or no file was saved.
///
/// # Errors
///
/// Returns `PyErr` if yt-dlp download fails or Python API call errors.
///
/// # Example
///
/// ```no_run
/// use melops_dl::{dl::download, asr::AudioFormat};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let (file_path, info) = download(
///     "https://youtube.com/watch?v=BaW_jenozKc",
///     AudioFormat::Pcm16.into()
/// )?;
///
/// if let Some(path) = file_path {
///     println!("Downloaded '{}' to: {}", info.title, path.display());
/// }
/// # Ok(())
/// # }
/// ```
pub fn download(
    url: &str,
    params: DownloadParams,
) -> Result<(Option<PathBuf>, DownloadInfo), PyErr> {
    Python::attach(|py| {
        let module = PyModule::from_code(py, c_str!(include_str!("./dl.py")), c"dl.py", c"dl")?;

        let py_params = params.into_pyobject(py)?;

        module
            .getattr("download")?
            .call1((url, py_params))?
            .extract()
    })
}
