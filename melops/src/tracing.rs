//! Tracing initialization for melops CLI and tests.

use eyre::Result;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

/// Initialize tracing subscriber
///
/// Returns a guard that must be kept alive for the duration of the program.
/// Dropping the guard will flush any remaining logs.
pub fn init() -> Result<WorkerGuard> {
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
