//! A byte-level Byte-Pair Encoding (BPE) tokenizer, written from scratch.
//!
//! BPE starts from raw bytes (256 possible values) and repeatedly merges the
//! most frequent adjacent pair into a new token, growing the vocabulary until
//! it reaches a target size. Because it operates on bytes, it can represent any
//! UTF-8 text -- English, Hindi, Tamil, emoji -- with no "unknown token".

use std::collections::HashMap;

/// An ordered merge rule: the pair (a, b) becomes the new token id.
pub type Pair = (u32, u32);

#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// Learned merges: pair -> new token id. The *rank* (insertion order) is the
    /// new id minus 256, and lower rank means "merge earlier".
    pub merges: HashMap<Pair, u32>,
    /// token id -> the raw bytes it expands to. Ids 0..=255 are the bytes.
    pub vocab: HashMap<u32, Vec<u8>>,
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer {
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        for b in 0u32..256 {
            vocab.insert(b, vec![b as u8]);
        }
        Tokenizer { merges: HashMap::new(), vocab }
    }

    /// Count how often each adjacent pair appears in `ids`.
    fn pair_counts(ids: &[u32]) -> HashMap<Pair, usize> {
        let mut counts = HashMap::new();
        for w in ids.windows(2) {
            *counts.entry((w[0], w[1])).or_insert(0) += 1;
        }
        counts
    }

    /// Replace every occurrence of `pair` in `ids` with `new_id`.
    fn merge(ids: &[u32], pair: Pair, new_id: u32) -> Vec<u32> {
        let mut out = Vec::with_capacity(ids.len());
        let mut i = 0;
        while i < ids.len() {
            if i + 1 < ids.len() && ids[i] == pair.0 && ids[i + 1] == pair.1 {
                out.push(new_id);
                i += 2;
            } else {
                out.push(ids[i]);
                i += 1;
            }
        }
        out
    }

    /// Train BPE on `text` until the vocabulary reaches `vocab_size` (>= 256).
    pub fn train(&mut self, text: &str, vocab_size: usize) {
        assert!(vocab_size >= 256, "vocab_size must be at least 256");
        let num_merges = vocab_size - 256;
        let mut ids: Vec<u32> = text.bytes().map(|b| b as u32).collect();

        for i in 0..num_merges {
            let counts = Self::pair_counts(&ids);
            // pick the most frequent pair; ties broken by the smaller pair for determinism
            let best = counts
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)));
            let (&pair, &count) = match best {
                Some(x) => x,
                None => break,
            };
            if count < 2 {
                break; // nothing worth merging
            }
            let new_id = 256 + i as u32;
            ids = Self::merge(&ids, pair, new_id);
            self.merges.insert(pair, new_id);
            let mut bytes = self.vocab[&pair.0].clone();
            bytes.extend_from_slice(&self.vocab[&pair.1]);
            self.vocab.insert(new_id, bytes);
        }
    }

    /// Encode text into token ids using the learned merges.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids: Vec<u32> = text.bytes().map(|b| b as u32).collect();
        while ids.len() >= 2 {
            // find the present pair with the lowest merge rank (earliest learned)
            let counts = Self::pair_counts(&ids);
            let mut best_pair: Option<Pair> = None;
            let mut best_rank = u32::MAX;
            for pair in counts.keys() {
                if let Some(&id) = self.merges.get(pair) {
                    if id < best_rank {
                        best_rank = id;
                        best_pair = Some(*pair);
                    }
                }
            }
            match best_pair {
                Some(pair) => ids = Self::merge(&ids, pair, self.merges[&pair]),
                None => break,
            }
        }
        ids
    }

    /// Decode token ids back into the original string.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for id in ids {
            if let Some(b) = self.vocab.get(id) {
                bytes.extend_from_slice(b);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

#[cfg(feature = "python")]
mod py;
