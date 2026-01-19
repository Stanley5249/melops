//! Mel CLI - Audio captioning tool

use clap::Parser;
use eyre::Result;
use melops::cli::{Cli, run};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    color_eyre::install()?;

    let _guard = melops::tracing::init()?;

    melops::ort::init()?;

    run(Cli::parse()).await
}
