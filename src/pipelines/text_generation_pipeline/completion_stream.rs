use futures::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Convenience wrapper around the streaming output of [`TextGenerationPipeline`].
pub struct CompletionStream<'a> {
    inner: Pin<Box<dyn Stream<Item = anyhow::Result<String>> + Send + 'a>>,
}

impl<'a> CompletionStream<'a> {
    pub(crate) fn new(inner: Pin<Box<dyn Stream<Item = anyhow::Result<String>> + Send + 'a>>) -> Self {
        Self { inner }
    }

    /// Collect the entire stream into a single `String`.
    pub async fn collect(mut self) -> anyhow::Result<String> {
        use futures::StreamExt;
        let mut out = String::new();
        while let Some(chunk) = self.inner.next().await {
            out.push_str(&chunk?);
        }
        Ok(out)
    }

    /// Take up to `n` chunks from the stream.
    ///
    /// If the underlying stream ends before `n` chunks are yielded,
    /// the returned vector will contain fewer elements.
    pub async fn take(mut self, n: usize) -> anyhow::Result<Vec<String>> {
        use futures::StreamExt;
        let mut out = Vec::new();
        for _ in 0..n {
            match self.inner.next().await {
                Some(chunk) => out.push(chunk?),
                None => break,
            }
        }
        Ok(out)
    }
}

impl<'a> Stream for CompletionStream<'a> {
    type Item = anyhow::Result<String>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut().inner.as_mut().poll_next(cx)
    }
}
