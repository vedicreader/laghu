# laghu-rs

The optional Rust core of [laghu](https://github.com/vedicreader/laghu). laghu loads model2vec static embedding models with far less memory than model2vec and gives the same vectors. With laghu-rs installed, laghu runs normalization, pre-tokenization, the Unigram Viterbi search and mean pooling in Rust threads, over the same memory-mapped files. It also tokenizes BERT WordPiece models (a BertNormalizer, a BertPreTokenizer and a WordPiece model) in Rust, with a port of `FastWordPiece` from [model2vec-rs](https://github.com/MinishLab/model2vec-rs), MIT, copyright (c) 2025 The Minish Lab (`LICENSE-model2vec-rs`).

``` sh
pip install laghu-rs     # or: pip install 'laghu[rust]'
```

Nothing else changes: laghu finds laghu-rs and uses it for the models it supports. Output is identical, bit for bit, to laghu's pure-Python path and to model2vec 0.9.

500-character chunks on the 14 cores of an M3 Max, best of three runs of `laghu.bench.speed`, measured on 2026-10-24 at a load average of 6.5 to 9. potion-multilingual-128M (Unigram):

| path | chunks/s | private memory |
|---|---|---|
| laghu with laghu-rs | 42,385 | 72 MB |
| laghu, pure Python | 1,940 | 67 MB |
| model2vec 0.9 | 21,105 | 1,369 MB |

potion-base-8M (WordPiece):

| path | chunks/s | private memory |
|---|---|---|
| laghu with laghu-rs | 432,938 | 81 MB |
| laghu with `tokenizers` | 55,514 | 93 MB |
| model2vec 0.9 | 55,600 | 140 MB |

The figure laghu-rs 0.1.0 gave for model2vec, 5,276 chunks/s, was measured with `tokenizers` on one core.

laghu falls back to pure Python when laghu-rs is not installed or `LAGHU_RUST=0` is set. It keeps Python's normalization and pre-tokenization when the `tokenizers` crate inside laghu-rs is not the same version as the installed Python `tokenizers`, or when the two disagree on a set of probe texts. It tokenizes a WordPiece model with `tokenizers` under the same two conditions, and for any pipeline the port does not reproduce.

laghu-rs needs Python 3.11 or later. Building it from source needs Rust and a C compiler.
