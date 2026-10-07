//! Train BPE on a corpus file: `cargo run --release --example train -- corpus.txt 4000`
use blazetok::Tokenizer;
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args.get(1).expect("usage: train <corpus.txt> <vocab_size>");
    let vocab_size: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(4000);

    let text = fs::read_to_string(path).expect("could not read corpus");
    println!("corpus: {} bytes", text.len());

    let mut tok = Tokenizer::new();
    tok.train(&text, vocab_size);
    println!("trained vocab size: {}", tok.vocab_size());

    let sample = "the quick brown fox";
    let ids = tok.encode(sample);
    println!("encode({sample:?}) -> {} tokens", ids.len());
    println!("roundtrip ok: {}", tok.decode(&ids) == sample);
}
