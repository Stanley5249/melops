//! FFmpeg child-process audio source.

use crate::segment::{SegmentConfig, SegmentStream};
use async_stream::stream;
use futures::Stream;
use std::path::PathBuf;
use std::process::{ExitStatus, Stdio};
use tokio::io::{AsyncRead, AsyncReadExt, BufReader};
use tokio::process::{Child, Command};

pub struct Ffmpeg {
    pub path: PathBuf,
    pub child: Child,
}

impl Ffmpeg {
    pub async fn new(path: PathBuf) -> std::io::Result<Self> {
        let child = Command::new("ffmpeg")
            .arg("-i")
            .arg(&path)
            .arg("-vn")
            .arg("-ac")
            .arg("1")
            .arg("-ar")
            .arg("16000")
            .arg("-acodec")
            .arg("pcm_f32le")
            .arg("-f")
            .arg("f32le")
            .arg("-")
            .stdout(Stdio::piped())
            .kill_on_drop(true)
            .spawn()?;

        Ok(Self { path, child })
    }

    pub fn reader(&mut self) -> Option<impl SegmentStream> {
        self.child.stdout.take().map(|io| BufReader::new(io))
    }

    pub async fn join(&mut self) -> std::io::Result<ExitStatus> {
        self.child.wait().await
    }
}

impl<T> SegmentStream for T
where
    T: AsyncRead + Send + Unpin,
{
    fn stream(
        &mut self,
        config: SegmentConfig,
    ) -> impl Stream<Item = std::io::Result<Vec<f32>>> + Send {
        let mut segment = Vec::with_capacity(config.window_size());

        stream! {
            loop {
                match self.read_f32_le().await {
                    Ok(data) => segment.push(data),
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                }

                if segment.len() == config.window_size() {
                    yield Ok(segment.clone());
                    segment.drain(0..config.step_size());
                }
            }

            if !segment.is_empty() {
                yield Ok(segment);
            }
        }
    }
}
