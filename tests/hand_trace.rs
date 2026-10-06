//! The primer's hand-traced example, checked against the real implementation.
use blazetok::Tokenizer;

#[test]
fn hand_trace_lower_lowest_slow_slowest() {
    let mut t = Tokenizer::new();
    t.train("lower lowest slow slowest", 256 + 6);
    let learned: Vec<String> = (256..262)
        .map(|id| String::from_utf8(t.vocab[&id].clone()).unwrap())
        .collect();
    assert_eq!(learned, ["lo", "low", "lowe", " s", "st", "lowest"]);
    assert_eq!(t.encode("lower lowest slow slowest").len(), 8);
}
