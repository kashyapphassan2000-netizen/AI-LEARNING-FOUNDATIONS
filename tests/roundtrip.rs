use blazetok::Tokenizer;

const CORPUS: &str = "\
the quick brown fox jumps over the lazy dog. the dog sleeps. \
the quick fox runs and the lazy dog runs too. \
नमस्ते दुनिया। यह एक छोटा वाक्य है। यह दुनिया सुंदर है। नमस्ते नमस्ते। \
வணக்கம் உலகம். இது ஒரு சிறிய வாக்கியம். இந்த உலகம் அழகானது. வணக்கம் வணக்கம். ";

#[test]
fn roundtrip_english() {
    let mut t = Tokenizer::new();
    t.train(CORPUS, 400);
    let s = "the quick brown fox";
    assert_eq!(t.decode(&t.encode(s)), s);
}

#[test]
fn roundtrip_hindi() {
    let mut t = Tokenizer::new();
    t.train(CORPUS, 400);
    let s = "नमस्ते दुनिया";
    assert_eq!(t.decode(&t.encode(s)), s);
}

#[test]
fn roundtrip_tamil() {
    let mut t = Tokenizer::new();
    t.train(CORPUS, 400);
    let s = "வணக்கம் உலகம்";
    assert_eq!(t.decode(&t.encode(s)), s);
}

#[test]
fn roundtrip_unseen_text() {
    // even text never seen in training must round-trip, thanks to byte fallback
    let mut t = Tokenizer::new();
    t.train(CORPUS, 400);
    let s = "zany ☃ pangram 12345 ✓";
    assert_eq!(t.decode(&t.encode(s)), s);
}

#[test]
fn training_grows_vocab() {
    let mut t = Tokenizer::new();
    assert_eq!(t.vocab_size(), 256);
    t.train(CORPUS, 400);
    assert!(t.vocab_size() > 256 && t.vocab_size() <= 400);
}

#[test]
fn merges_reduce_token_count() {
    let mut t = Tokenizer::new();
    t.train(CORPUS, 400);
    let s = "the quick brown fox jumps over the lazy dog";
    let raw = s.bytes().count();
    let encoded = t.encode(s).len();
    assert!(encoded < raw, "BPE should produce fewer tokens than raw bytes");
}
