# laghu-rs

The optional Rust core of [laghu](https://github.com/vedicreader/laghu). laghu loads model2vec static embedding models with far less memory than model2vec and gives the same vectors. With laghu-rs installed, laghu runs normalization, pre-tokenization, the Unigram Viterbi search and mean pooling in Rust threads, over the same memory-mapped files.

``` sh
pip install laghu-rs     # or: pip install 'laghu[rust]'
```

Nothing else changes: laghu finds laghu-rs and uses it for the models it supports. Output is identical, bit for bit, to laghu's pure-Python path and to model2vec 0.9.

On potion-multilingual-128M, 500-character chunks on 14 cores of an M3 Max:

| path | chunks/s | private memory |
|---|---|---|
| laghu with laghu-rs | 39,750 | 64 MB |
| laghu, pure Python | 1,736 | 67 MB |
| model2vec 0.9 | 5,276 | 1,360 MB |

laghu falls back to pure Python when laghu-rs is not installed or `LAGHU_RUST=0` is set. It keeps Python's normalization and pre-tokenization when the `tokenizers` crate inside laghu-rs is not the same version as the installed Python `tokenizers`, or when the two disagree on a set of probe texts.

laghu-rs needs Python 3.11 or later. Building it from source needs Rust and a C compiler.
