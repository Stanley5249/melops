//! yt-dlp download function and Python API wrapper.
//!
//! Main entry point for downloading media via yt-dlp.
//!
//! ```no_run
//! use melops_dl::dl::download;
//! use melops_dl::asr::AudioFormat;
//! use tokio::sync::broadcast;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let (tx, _rx) = broadcast::channel(10);
//! let info = download("https://youtube.com/watch?v=example", AudioFormat::Pcm16.into(), tx)?;
//! println!("Downloaded '{}'", info.title);
//! # Ok(())
//! # }
//! ```

use crate::info::DownloadInfo;
use crate::params::DownloadParams;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyDict, PyList, PyTuple};
use std::path::PathBuf;
use tokio::sync::broadcast;

/// Downloads media from a single URL using yt-dlp.
///
/// This is the main API that accepts a tokio `broadcast::Sender<PathBuf>` to broadcast file paths
/// from yt-dlp's post_hooks. The sender will be called by Python for each file created
/// during post-processing. Multiple subscribers can receive the same paths simultaneously.
///
/// Returns `DownloadInfo` metadata. File paths are broadcast through the provided sender.
///
/// # Errors
///
/// Returns `PyErr` if yt-dlp download fails or Python API call errors.
///
/// # Example
///
/// ```no_run
/// use melops_dl::{dl::download, asr::AudioFormat};
/// use tokio::sync::broadcast;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let (tx, mut rx) = broadcast::channel(10);
/// let info = download("https://youtube.com/watch?v=BaW_jenozKc", AudioFormat::Pcm16.into(), tx)?;
///
/// // Paths are broadcast as each file completes
/// while let Ok(path) = rx.blocking_recv() {
///     println!("Downloaded to: {}", path.display());
/// }
/// # Ok(())
/// # }
/// ```
pub fn download(
    url: &str,
    params: DownloadParams,
    tx: broadcast::Sender<PathBuf>,
) -> Result<DownloadInfo, PyErr> {
    // Create weak sender for Python closure (prevents holding channel open)
    let weak_tx = tx.downgrade();

    Python::attach(|py| {
        // Load the Python download module
        let module = PyModule::from_code(py, c_str!(include_str!("./dl.py")), c"dl.py", c"dl")?;

        // Create a Rust closure that broadcasts file paths through the channel
        // Uses WeakSender to avoid holding channel open indefinitely
        let callback = PyCFunction::new_closure(
            py,
            Some(c"callback"),
            None,
            move |args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>| -> PyResult<()> {
                // yt-dlp passes the file path as the first (and only) argument
                let file_path: PathBuf = args.get_item(0)?.extract()?;

                // Upgrade weak sender and broadcast (fails silently if no receivers)
                if let Some(tx) = weak_tx.upgrade() {
                    tx.send(file_path).ok();
                }

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

        // Call the Python download function with single URL
        let result: DownloadInfo = module
            .getattr("download")?
            .call1((url, params_dict))?
            .extract()?;

        Ok(result)
    })
}
