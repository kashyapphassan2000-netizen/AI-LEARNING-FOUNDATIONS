# AI Learning Foundations

Weeks 1–5 of the ZERO_TO_ANTHROPIC 12-project roadmap, each built step by step,
day by day, with real measured results. Every project is a **standalone repo** with its
own history (one commit per day), its own README telling the build story day by day,
tests, and CI.

| # | project | what it is | headline result (measured) |
|---|---|---|---|
| 01 | [tiny-vec](tiny-vec/) | Vector/Matrix, Strassen, reverse-mode autograd, SVD/PCA in pure Python | NumPy ~700x faster than naive matmul; Strassen wins from n≈256 (1.65x at 512); PCA matches scikit-learn to 1e-10; 11 tests, 80% coverage |
| 02 | [blazetok](blazetok/) | byte-level BPE tokenizer in Rust + PyO3 Python bindings | round-trips English/Hindi/Tamil/emoji; on held-out text 1.33x (Hindi) / 1.86x (Tamil) fewer tokens than GPT-4's cl100k; 7 Rust tests |
| 03 | [micrograd-fast](micrograd-fast/) | scalar + NumPy tensor autograd, MLP, Adam, fused softmax-xent | **real MNIST: 96.85% in 6.0 s** on CPU; scalar→tensor 410x cold / ~4200x warm; gradient checks catch injected bugs |
| 04 | [kalman-rag](kalman-rag/) | hybrid RAG (TF-IDF/BGE-M3 + BM25 + RRF) with cited Claude answers | 90% top-1 on goldset; ablation shows BM25 alone = 100% (hybrid adds nothing with two lexical retrievers); 20% on paraphrases |
| 05 | [skew-detector](skew-detector/) | training-serving drift CLI: schema, KS+Bonferroni, PSI, chi², CI gate | catches all 4 injected drifts, 0 false positives on controls; exit 1 on drift / 0 on clean, enforced in CI |

## What was not done (needs your accounts or keys)

- **TestPyPI uploads** (tiny-vec, blazetok): need your TestPyPI API token. Exact commands are in each README.
- **Live Claude/Cohere calls** (kalman-rag): no API keys in the build environment. The answer
  step is verified with fake clients; run it with `ANTHROPIC_API_KEY` / `COHERE_API_KEY` set.
- **BGE-M3 + pgvector** (kalman-rag): reference code, compile-checked, not executed (multi-GB model + Postgres).
- **Hugging Face Space deploy** and the **Loom recording** (micrograd-fast): yours to do.
- **CI badges**: GitHub only runs workflows from a repo's root `.github/`, so each project's CI
  starts running once it lives in its own repo (next section).

## Splitting into five separate repos

Each folder is a `git subtree` import that preserves the project's full history; splitting
reproduces the original commits exactly (verified: same hashes). Create five **empty**
repos on GitHub (`tiny-vec`, `blazetok`, `micrograd-fast`, `kalman-rag`, `skew-detector`),
then:

```bash
git clone https://github.com/kashyapphassan2000-netizen/AI-LEARNING-FOUNDATIONS
cd AI-LEARNING-FOUNDATIONS
git checkout claude/relaxed-allen-3wwpiq
for r in tiny-vec blazetok micrograd-fast kalman-rag skew-detector; do
  git subtree split --prefix=$r -b split-$r
  git push https://github.com/kashyapphassan2000-netizen/$r.git split-$r:main
done
git push https://github.com/kashyapphassan2000-netizen/micrograd-fast.git split-micrograd-fast:refs/tags/v0.1.0
```
