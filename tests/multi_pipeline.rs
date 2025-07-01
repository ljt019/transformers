use transformers::pipelines::global_cache;
use transformers::pipelines::text_generation_pipeline::*;

#[test]
fn multiple_pipelines_share_weights_and_have_independent_caches() -> anyhow::Result<()> {
    // Ensure the cache is clean so we can accurately count models
    global_cache().clear();

    let mut pipelines = Vec::new();
    for _ in 0..10 {
        let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
            .temperature(0.7)
            .max_len(10)
            .build()?;
        pipelines.push(pipeline);
    }

    // Only one model should be loaded
    assert_eq!(global_cache().len(), 1);

    let prompt = "Hello, world!";
    let _ = pipelines[0].prompt_completion(prompt)?;

    // The first pipeline should have advanced its context
    assert!(pipelines[0].context_position() > 0);

    // Other pipelines should remain untouched
    for (idx, p) in pipelines.iter().enumerate().skip(1) {
        assert_eq!(
            p.context_position(),
            0,
            "pipeline {} reused context",
            idx + 1
        );
    }

    Ok(())
}
