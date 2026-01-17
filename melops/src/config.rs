//! Configuration types for resolved CLI arguments.
//!
//! This module contains Config structs and their TryFrom implementations.
//! Args structs (for CLI parsing) remain in cli.rs.

use crate::cli::{ModelArgs, ModelSource};
use eyre::Result;
use hf_hub::Cache;
use hf_hub::api::sync::Api;
use melops_asr::models::tdt::core::TdtModel;
use melops_asr::types::ModelRepo;
use std::path::PathBuf;

/// Resolved model configuration.
///
/// Converted from ModelArgs via TryFrom.
/// Contains the ModelRepo enum ready for use.
#[derive(Debug)]
pub struct ModelConfig {
    pub repo: ModelRepo,
}

impl TryFrom<ModelArgs> for ModelConfig {
    type Error = eyre::Error;

    fn try_from(args: ModelArgs) -> Result<Self> {
        let repo = match args.model_source {
            ModelSource::Auto => {
                let path = PathBuf::from(&args.model_id);
                if path.is_dir() {
                    ModelRepo::Path(path)
                } else {
                    let api = Api::new()?;
                    ModelRepo::Api(api.model(args.model_id))
                }
            }
            ModelSource::Path => ModelRepo::Path(PathBuf::from(args.model_id)),
            ModelSource::Cache => ModelRepo::Cache(Cache::from_env().model(args.model_id)),
            ModelSource::Api => ModelRepo::Api(Api::new()?.model(args.model_id)),
        };

        Ok(Self { repo })
    }
}

impl ModelConfig {
    /// Load TDT model with ONNX Runtime session
    pub fn load(&self) -> Result<TdtModel> {
        let builder = crate::ort::build_session()?;
        TdtModel::from_repo(&self.repo, builder)
    }
}
