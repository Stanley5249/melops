//! yt-dlp download function and Python API wrapper.
//!
//! Main entry point for downloading media via yt-dlp.
//!
//! ```no_run
//! use melops_dl::{dl::download, asr::AudioFormat};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let (file_paths, info) = download("https://youtube.com/watch?v=example", AudioFormat::Pcm16.into())?;
//! println!("Downloaded '{}' to {:?}", info.title, file_paths);
//! # Ok(())
//! # }
//! ```

use crate::info::DownloadInfo;
use crate::params::DownloadParams;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyDict, PyList, PyTuple};
use std::path::PathBuf;
use std::sync::mpsc;

/// Downloads media from URL using yt-dlp with a custom sender for file paths.
///
/// This is the low-level API that accepts an `mpsc::Sender<String>` to receive file paths
/// from yt-dlp's post_hooks. The sender will be moved into a closure that gets called by
/// Python for each file created during post-processing.
///
/// Returns `DownloadInfo` metadata. File paths are sent through the provided sender.
///
/// # Errors
///
/// Returns `PyErr` if yt-dlp download fails or Python API call errors.
///
/// # Example
///
/// ```no_run
/// use melops_dl::{dl::download_with_sender, asr::AudioFormat};
/// use std::sync::mpsc;
/// use std::thread;
/// use std::time::Duration;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let (tx, rx) = mpsc::channel();
/// let info = download_with_sender(
///     "https://youtube.com/watch?v=BaW_jenozKc",
///     AudioFormat::Pcm16.into(),
///     tx
/// )?;
///
/// // Note: The sender is held by Python's closure and won't drop immediately.
/// // Use try_iter() or recv_timeout() instead of blocking iteration.
/// thread::sleep(Duration::from_millis(100));
/// for file_path in rx.try_iter() {
///     println!("Downloaded '{}' to: {}", info.title, file_path.display());
/// }
/// # Ok(())
/// # }
/// ```
pub fn download_with_sender(
    url: &str,
    params: DownloadParams,
    tx: mpsc::Sender<PathBuf>,
) -> Result<DownloadInfo, PyErr> {
    Python::attach(|py| {
        // Load the Python download module
        let module = PyModule::from_code(py, c_str!(include_str!("./dl.py")), c"dl.py", c"dl")?;

        // Create a Rust closure that sends file paths through the channel
        // The sender is moved into this closure and will be called by yt-dlp's post_hooks
        let callback = PyCFunction::new_closure(
            py,
            Some(c"callback"),
            None,
            move |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| -> PyResult<()> {
                // yt-dlp passes the file path as the first (and only) argument
                let file_path: PathBuf = args.get_item(0)?.extract()?;

                // Send the file path through the channel
                // Ignore send errors (receiver might be dropped if download was cancelled)
                tx.send(file_path).ok();

                Ok(())
            },
        )?;

        // Convert params to Python object
        let params_obj = params.into_pyobject(py)?;
        let params_dict = params_obj.cast::<PyDict>()?;

        // Get existing post_hooks list or create a new one
        match params_dict.get_item("post_hooks")? {
            Some(hooks) => hooks.cast::<PyList>()?.append(callback)?,
            None => params_dict.set_item("post_hooks", PyList::new(py, [callback])?)?,
        };

        // Call the Python download function
        module
            .getattr("download")?
            .call1((url, params_dict))?
            .extract()
    })
}

/// Downloads media from URL using yt-dlp (convenience wrapper).
///
/// This is a convenience function that creates the channel internally and collects
/// all file paths into a `Vec<PathBuf>`. For more control over path collection,
/// use `download_with_sender` directly.
///
/// Returns `(file_paths, info)` where `file_paths` contains all file paths created during
/// post-processing (may include multiple files if yt-dlp creates multiple outputs).
/// The vec will be empty if download failed or no files were saved.
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
/// let (file_paths, info) = download(
///     "https://youtube.com/watch?v=BaW_jenozKc",
///     AudioFormat::Pcm16.into()
/// )?;
///
/// for path in file_paths {
///     println!("Downloaded '{}' to: {}", info.title, path.display());
/// }
/// # Ok(())
/// # }
/// ```
pub fn download(url: &str, params: DownloadParams) -> Result<(Vec<PathBuf>, DownloadInfo), PyErr> {
    // Create channel for receiving file paths from Python post_hooks
    let (tx, rx) = mpsc::channel();

    // Call the low-level API with the cloned sender
    let info = download_with_sender(url, params, tx)?;

    // Collect all file paths sent through the channel
    let file_paths: Vec<PathBuf> = rx.try_iter().collect();

    Ok((file_paths, info))
}
