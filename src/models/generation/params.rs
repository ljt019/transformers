/// Generation parameters for language models.
#[derive(Debug, Clone)]
pub struct GenerationParams {
    pub temperature: f64,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
    pub max_len: usize,
    pub top_p: f64,   // 0.0..=1.0 ; 0 or 1 means disabled
    pub top_k: usize, // 0 means disabled
    pub min_p: f64,   // 0.0..=1.0 ; 0 means disabled
}

impl GenerationParams {
    pub fn new(
        temperature: f64,
        repeat_penalty: f32,
        repeat_last_n: usize,
        seed: u64,
        max_len: usize,
        top_p: f64,
        top_k: usize,
        min_p: f64,
    ) -> Self {
        Self {
            temperature,
            repeat_penalty,
            repeat_last_n,
            seed,
            max_len,
            top_p,
            top_k,
            min_p,
        }
    }
}