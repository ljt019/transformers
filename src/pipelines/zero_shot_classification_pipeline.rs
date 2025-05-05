//!
//! Zero-Shot Text Classification Pipeline using ModernBERT with Natural Language Inference (NLI).
//!
//! This pipeline classifies text sequences according to user-provided candidate labels
//! without requiring labeled data for those specific labels during training.
//! It leverages a ModernBERT model fine-tuned on an NLI task.
//!
//! The core idea is to frame the classification task as an NLI problem.
//! For each `(premise, candidate_label)` pair, a hypothesis is constructed (e.g.,
//! "This example is {candidate_label}."). The NLI model then predicts the probability
//! that the premise entails the hypothesis. The label with the highest entailment probability
//! is chosen as the prediction.
//!
//! ## Example
//!
//! ```rust,no_run
//! use transformers::pipelines::zero_shot_classification_pipeline::{
//!     ZeroShotClassificationPipelineBuilder,
//!     ZeroShotModernBertSize
//! };
//! # use anyhow::Result;
//!
//! # fn main() -> Result<()> {
//! // Use the builder to specify the model size
//! let pipeline = ZeroShotClassificationPipelineBuilder::new(
//!     ZeroShotModernBertSize::Large // Currently only Large is supported for zero-shot
//! ).build()?;
//!
//! let premise = "The Federal Reserve raised interest rates citing inflationary pressures.";
//! let candidate_labels = &["economics", "politics", "environment"];
//!
//! // Predict returns a sorted list of labels and their scores
//! let results = pipeline.predict(premise, candidate_labels)?;
//!
//! println!("Premise: {}", premise);
//! println!("Candidate Labels: {:?}", candidate_labels);
//! println!("Predictions:");
//! for (label, score) in results {
//!     println!("  - {}: {:.4}", label, score);
//! }
//! // Example Output:
//! // Premise: The Federal Reserve raised interest rates citing inflationary pressures.
//! // Candidate Labels: ["economics", "politics", "environment"]
//! // Predictions:
//! //   - economics: 0.9523
//! //   - politics: 0.0391
//! //   - environment: 0.0086
//! # Ok(())
//! # }
//! ```

use crate::models::zero_shot_modern_bert::ZeroShotModernBertModel;
use anyhow::Result;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

// Re-export the size enum for easier use by consumers of the pipeline
/// Available sizes for the Zero-Shot ModernBERT model used in the pipeline.
pub use crate::models::zero_shot_modern_bert::ZeroShotModernBertSize;

/// Builder for configuring and constructing a zero-shot classification pipeline.
///
/// Call `.build()` to obtain a `ZeroShotClassificationPipeline`.
#[derive(Debug)]
pub struct ZeroShotClassificationPipelineBuilder {
    size: ZeroShotModernBertSize,
}

impl ZeroShotClassificationPipelineBuilder {
    /// Creates a new builder for the specified ModernBERT zero-shot model size.
    ///
    /// Currently, only `ZeroShotModernBertSize::Large` is well-supported.
    pub fn new(size: ZeroShotModernBertSize) -> Self {
        Self { size }
    }

    /// Constructs the `ZeroShotClassificationPipeline`.
    ///
    /// This involves downloading model and tokenizer assets (if not cached) and
    /// loading them.
    pub fn build(self) -> Result<ZeroShotClassificationPipeline> {
        // Use the model's helper function to get the correct repo ID
        let model_id = ZeroShotModernBertModel::get_tokenizer_repo_info(self.size);

        // Fetch tokenizer first using the derived model_id
        let api = Api::new()?;
        let repo = api.repo(Repo::new(model_id.clone(), RepoType::Model));

        let tokenizer_filename = repo.get("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(&tokenizer_filename).map_err(|e| {
            anyhow::Error::msg(format!(
                "Failed to load tokenizer from {:?}: {}",
                tokenizer_filename, e
            ))
        })?;
        // Note: Padding configuration is handled *inside* the model's predict method
        // because it needs to pad based on the *batch* of premise-hypothesis pairs.

        // Build the ZeroShotModernBertModel using the specified size
        let model = ZeroShotModernBertModel::new(self.size)?;

        Ok(ZeroShotClassificationPipeline { model, tokenizer })
    }
}

/// A ready-to-use pipeline for zero-shot text classification using ModernBERT.
///
/// Built using `ZeroShotClassificationPipelineBuilder`.
/// Call `.predict(premise, candidate_labels)` to classify text.
pub struct ZeroShotClassificationPipeline {
    model: ZeroShotModernBertModel,
    tokenizer: Tokenizer,
}

impl ZeroShotClassificationPipeline {
    /// Classifies a given text (`premise`) according to a set of `candidate_labels`.
    ///
    /// Returns a `Vec` of `(String, f32)` tuples, where each tuple contains a
    /// candidate label and its associated score (probability of entailment), sorted
    /// in descending order of score.
    ///
    /// # Arguments
    ///
    /// * `premise` - The text sequence to classify.
    /// * `candidate_labels` - A slice of string slices, representing the potential categories.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or model inference fails.
    pub fn predict(&self, premise: &str, candidate_labels: &[&str]) -> Result<Vec<(String, f32)>> {
        // Delegate the core logic to the underlying model's predict method
        self.model
            .predict(&self.tokenizer, premise, candidate_labels)
    }
}
