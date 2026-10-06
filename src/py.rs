//! Python bindings via PyO3. Compiled only with `--features python`.
//! This wraps the pure-Rust `Tokenizer` so Python can call into it directly.

use pyo3::prelude::*;
use crate::Tokenizer as CoreTokenizer;

#[pyclass(name = "Tokenizer")]
struct PyTokenizer {
    inner: CoreTokenizer,
}

#[pymethods]
impl PyTokenizer {
    #[new]
    fn new() -> Self {
        PyTokenizer { inner: CoreTokenizer::new() }
    }

    fn train(&mut self, text: &str, vocab_size: usize) {
        self.inner.train(text, vocab_size);
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    fn decode(&self, ids: Vec<u32>) -> String {
        self.inner.decode(&ids)
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}

#[pymodule]
fn blazetok(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    Ok(())
}
