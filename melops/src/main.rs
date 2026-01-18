//! Mel CLI - Audio captioning tool

use clap::Parser;
use eyre::Result;
use melops::cli::{Cli, run};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

fn init_tracing() -> Result<WorkerGuard> {
    let subscriber = tracing_subscriber::registry();

    #[cfg(feature = "console")]
    let subscriber = {
        let layer = console_subscriber::ConsoleLayer::builder()
            .with_default_env()
            .spawn();

        subscriber.with(layer)
    };

    let (non_blocking, guard) = tracing_appender::non_blocking(std::io::stderr());

    let layer = tracing_subscriber::fmt::layer()
        .with_span_events(FmtSpan::CLOSE)
        .with_writer(non_blocking)
        .with_filter(EnvFilter::from_default_env());

    subscriber.with(layer).init();

    Ok(guard)
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    color_eyre::install()?;

    let _guard = init_tracing()?;

    melops::ort::init()?;

    run(Cli::parse()).await
}
