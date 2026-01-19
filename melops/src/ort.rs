//! ONNX Runtime initialization and session configuration

use crate::cache::CacheDir;
use eyre::Result;
use ort::environment::GlobalThreadPoolOptions;
use ort::session::Session;
use ort::session::builder::SessionBuilder;

#[allow(unused_imports)]
use ort::ep::*;

/// Initialize ONNX Runtime with execution providers and global configuration
pub fn init() -> Result<bool> {
    let cache_dir = CacheDir::new(None)?;
    let _ort_cache = cache_dir.join("ort");

    let eps = [
        #[cfg(feature = "cuda")]
        CUDA::default().build(),
        #[cfg(feature = "tensorrt")]
        TensorRT::default().build(),
        #[cfg(feature = "openvino")]
        OpenVINO::default()
            .with_device_type("HETERO:GPU,CPU")
            .with_cache_dir(_ort_cache.to_string_lossy())
            .with_precision("FP16")
            .build(),
        #[cfg(feature = "directml")]
        DirectML::default().build(),
        #[cfg(feature = "coreml")]
        CoreML::default().build(),
    ];

    let thread_pool_options = GlobalThreadPoolOptions::default()
        .with_intra_threads(2)?
        .with_spin_control(true)?;

    let commit = ort::init()
        .with_name("melops")
        .with_execution_providers(eps)
        .with_global_thread_pool(thread_pool_options)
        .commit();

    Ok(commit)
}

/// Create session builder with optimizations configured
pub fn build_session() -> ort::Result<SessionBuilder> {
    let builder = Session::builder()?;

    // According to the OpenVINO documentation, disabling optimization can often yield better results
    #[cfg(feature = "openvino")]
    let builder =
        builder.with_optimization_level(ort::session::builder::GraphOptimizationLevel::Disable)?;

    Ok(builder)
}
