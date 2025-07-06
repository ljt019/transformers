use futures::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};
use pin_project_lite::pin_project;

/// Convenience wrapper around the streaming output of [`TextGenerationPipeline`].
pin_project! {
    pub struct CompletionStream<S> {
        #[pin]
        inner: Pin<Box<S>>,
    }
}

impl<S> CompletionStream<S> {
    pub(crate) fn new(inner: S) -> Self {
        Self { inner: Box::pin(inner) }
    }

    /// Collect the entire stream into a single `String`.
    pub async fn collect(mut self) -> anyhow::Result<String>
    where
        S: Stream<Item = anyhow::Result<String>>,
    {
        use futures::StreamExt;
        let mut out = String::new();
        while let Some(chunk) = self.inner.as_mut().next().await {
            out.push_str(&chunk?);
        }
        Ok(out)
    }

    /// Take up to `n` chunks from the stream.
    ///
    /// If the underlying stream ends before `n` chunks are yielded,
    /// the returned vector will contain fewer elements.
    pub async fn take(mut self, n: usize) -> anyhow::Result<Vec<String>>
    where
        S: Stream<Item = anyhow::Result<String>>,
    {
        use futures::StreamExt;
        let mut out = Vec::new();
        for _ in 0..n {
            match self.inner.as_mut().next().await {
                Some(chunk) => out.push(chunk?),
                None => break,
            }
        }
        Ok(out)
    }
}

impl<S> Stream for CompletionStream<S>
where
    S: Stream<Item = anyhow::Result<String>>, 
{
    type Item = anyhow::Result<String>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        this.inner.as_mut().as_mut().poll_next(cx)
    }
}
