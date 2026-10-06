"""Offline version of compare_tiktoken.py: no downloads needed.

Compares blazetok against the raw byte-level baseline -- one token per UTF-8
byte, which is what an English-first byte-level tokenizer degrades to on
scripts it never learned merges for. Also reports a *held-out* check on real
sentences that are not drawn from the word bank, so the headline number comes
with its honest caveat attached.
"""
import random
import blazetok
from compare_tiktoken import HINDI, TAMIL, make_text

HELD_OUT = {
    "Hindi": "यह एक छोटा वाक्य है। यह दुनिया सुंदर है। नमस्ते दुनिया।",
    "Tamil": "இது ஒரு சிறிய வாக்கியம். இந்த உலகம் அழகானது. வணக்கம் உலகம்.",
}


def main():
    random.seed(0)
    print(f"{'script':<8} {'UTF-8 bytes/char':>17} {'blazetok tok/char':>18} {'vs raw bytes':>13}")
    tokenizers = {}
    for name, bank in [("Hindi", HINDI), ("Tamil", TAMIL)]:
        train_text = make_text(bank, 6000)
        test_text = make_text(bank, 800)
        tok = blazetok.Tokenizer()
        tok.train(train_text, 2000)
        tokenizers[name] = tok
        n_bytes = len(test_text.encode("utf-8"))
        n_tok = len(tok.encode(test_text))
        chars = len(test_text)
        print(f"{name:<8} {n_bytes / chars:>17.2f} {n_tok / chars:>18.3f} "
              f"{n_bytes / n_tok:>11.0f}x")

    print("\nheld-out sentences (words/sentence structure partly unseen):")
    for name, text in HELD_OUT.items():
        tok = tokenizers[name]
        n_bytes = len(text.encode("utf-8"))
        n_tok = len(tok.encode(text))
        print(f"{name:<8} {n_bytes:4d} bytes -> {n_tok:4d} tokens "
              f"({n_bytes / n_tok:.1f}x fewer), roundtrip {tok.decode(tok.encode(text)) == text}")


if __name__ == "__main__":
    main()
