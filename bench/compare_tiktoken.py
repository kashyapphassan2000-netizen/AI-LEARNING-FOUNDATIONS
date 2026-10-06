"""Compare token efficiency (tokens per character) of an Indic-trained BPE
against tiktoken's cl100k (the GPT-4 tokenizer) on Hindi and Tamil text.

The corpus is procedurally generated from common-word banks so the demo runs
offline and reproducibly. The real project (per the roadmap) trains on
AI4Bharat / Shrutilipi data; the *direction* of the result is the same: tiktoken
was trained mostly on English and byte-falls-back on Devanagari/Tamil, spending
many tokens per character, while a BPE that has seen Indic text compresses it.
"""
import random
import blazetok
import tiktoken

random.seed(0)

HINDI = ["नमस्ते", "दुनिया", "पानी", "घर", "किताब", "स्कूल", "बच्चा", "सूरज",
           "चाँद", "पेड़", "नदी", "पहाड़", "रास्ता", "दोस्त", "खाना", "समय",
           "सुबह", "रात", "काम", "शहर", "गाँव", "हवा", "बारिश", "फूल", "आसमान"]
TAMIL = ["வணக்கம்", "உலகம்", "தண்ணீர்", "வீடு", "புத்தகம்", "பள்ளி", "குழந்தை",
           "சூரியன்", "நிலா", "மரம்", "ஆறு", "மலை", "பாதை", "நண்பன்", "உணவு",
           "நேரம்", "காலை", "இரவு", "வேலை", "நகரம்", "கிராமம்", "காற்று", "மழை"]
ENGLISH = ["world", "water", "house", "book", "school", "child", "sun", "moon",
             "tree", "river", "mountain", "road", "friend", "food", "time"]


def make_text(words, n):
    out = []
    for _ in range(n):
        out.append(random.choice(words))
        if random.random() < 0.12:
            out.append(random.choice(["।", ".", ","]))
    return " ".join(out)


def tokens_per_char(n_tokens, text):
    return n_tokens / max(1, len(text))


def main():
    cl100k = tiktoken.get_encoding("cl100k_base")

    for name, bank in [("Hindi", HINDI), ("Tamil", TAMIL), ("English", ENGLISH)]:
        train_text = make_text(bank, 6000)
        test_text = make_text(bank, 800)

        tok = blazetok.Tokenizer()
        tok.train(train_text, 2000)

        bz = len(tok.encode(test_text))
        tk = len(cl100k.encode(test_text))
        chars = len(test_text)

        print(f"\n=== {name} (test text: {chars} chars) ===")
        print(f"  blazetok (Indic-trained) : {bz:6d} tokens  "
              f"({tokens_per_char(bz, test_text):.3f} tok/char)")
        print(f"  tiktoken cl100k (GPT-4)  : {tk:6d} tokens  "
              f"({tokens_per_char(tk, test_text):.3f} tok/char)")
        ratio = tk / bz if bz else 0
        print(f"  -> tiktoken uses {ratio:.2f}x as many tokens on this text")


if __name__ == "__main__":
    main()
