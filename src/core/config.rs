#[derive(Clone)]
pub struct GenerationConfig {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u64>,
    pub min_p: Option<f64>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<usize>,
    pub eos_token_ids: Vec<u64>,
}
