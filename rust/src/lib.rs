//! The optional Rust core of laghu: Unigram Viterbi over byte offsets, searching a CSR byte trie mapped from laghu's
//! vocabulary file, and mean pooling over a mapped float32 table, batched over threads with the GIL released.
//! `Pre` normalizes, splits out added tokens and pre-tokenizes with the `tokenizers` crate, the version Python `tokenizers` has,
//! so whole texts can go to the threads. `WordPiece` gives the ids of the BERT WordPiece pipeline without the `tokenizers` crate's
//! offsets. Every result must equal laghu's Python path, and model2vec's, bit for bit.
use pyo3::buffer::{Element, PyBuffer};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokenizers::models::ModelWrapper;
use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::{Model, OffsetReferential, OffsetType, PreTokenizer, Tokenizer};
use unicode_categories::UnicodeCategories;
use unicode_normalization_alignments::UnicodeNormalization;

/// The `tokenizers` version pinned in Cargo.toml, which Python `tokenizers` must also be for `Pre` to be used.
const TOKENIZERS: &str = "0.23.2";

const NONE: u32 = u32::MAX;

/// A one-dimensional C-contiguous buffer held for as long as the slice over it is used.
struct Arr<T: Element> { _buf: PyBuffer<T>, ptr: usize, len: usize }

impl<T: Element> Arr<T> {
    fn get(o: &Bound<'_, PyAny>, name: &str) -> PyResult<Arr<T>> {
        let b = PyBuffer::<T>::get(o)?;
        if b.dimensions() != 1 || !b.is_c_contiguous() { return Err(PyValueError::new_err(format!("{name}: not a 1-D contiguous buffer"))); }
        let (ptr, len) = (b.buf_ptr() as usize, b.item_count());
        Ok(Arr { _buf: b, ptr, len })
    }
    #[inline(always)]
    fn s(&self) -> &[T] { unsafe { std::slice::from_raw_parts(self.ptr as *const T, self.len) } }
}

#[derive(IntoPyObject)]
enum Item { W(String), I(i64) }

enum Part<'a> { W(&'a str), I(i64) }

/// A text's normalization, added-token split and pre-tokenization, done as `TokenizerImpl::encode_single_sequence` does them,
/// by a `Tokenizer` whose model is a stub: `real` takes the stub's added-token ids to the ids in tokenizer.json.
#[pyclass(frozen)]
struct Pre { tok: Tokenizer, real: HashMap<u32, i64> }

impl Pre {
    /// `f` on each pre-token and each added token's id of `s`, in order.
    fn each(&self, s: &str, mut f: impl FnMut(Part) -> Result<(), String>) -> Result<(), String> {
        let mut pts = self.tok.get_added_vocabulary().extract_and_normalize(self.tok.get_normalizer(), s);
        if let Some(p) = self.tok.get_pre_tokenizer() { p.pre_tokenize(&mut pts).map_err(|e| e.to_string())?; }
        for (w, _, toks) in pts.get_splits(OffsetReferential::Normalized, OffsetType::None) {
            match toks {
                Some(ts) => for t in ts { f(Part::I(*self.real.get(&t.id).ok_or_else(|| format!("unknown added token id {}", t.id))?))?; },
                None => f(Part::W(w))?,
            }
        }
        Ok(())
    }
}

#[pymethods]
impl Pre {
    /// Build from tokenizer JSON `json` whose model is a stub, with `added` mapping each added token's content to its id.
    #[new]
    fn new(json: &str, added: HashMap<String, i64>) -> PyResult<Pre> {
        let tok = Tokenizer::from_str(json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut real = HashMap::new();
        for (c, k) in added {
            let s = tok.token_to_id(&c).ok_or_else(|| PyValueError::new_err(format!("added token {c:?} missing")))?;
            real.insert(s, k);
        }
        Ok(Pre { tok, real })
    }

    /// Text `s` as the Viterbi search receives it, as laghu's `Unigram.split`: pre-tokens (str) and added-token ids (int).
    fn split(&self, py: Python<'_>, s: &str) -> PyResult<Vec<Item>> {
        let mut out = Vec::new();
        py.detach(|| self.each(s, |p| { out.push(match p { Part::W(w) => Item::W(w.to_owned()), Part::I(k) => Item::I(k) }); Ok(()) }))
            .map_err(PyValueError::new_err)?;
        Ok(out)
    }
}

/// The texts of a batch: each already split by Python, or whole, for `Pre` to split in the thread that segments it.
enum Texts<'a> { Split(Vec<Vec<Item>>), Whole(Vec<String>, &'a Pre) }

impl Texts<'_> {
    fn len(&self) -> usize { match self { Texts::Split(v) => v.len(), Texts::Whole(v, _) => v.len() } }
}

#[inline(always)]
fn utf8_len(b: u8) -> usize { if b < 0x80 { 1 } else if b < 0xE0 { 2 } else if b < 0xF0 { 3 } else { 4 } }

/// A Unigram segmenter over the mapped vocabulary: the byte trie (edge starts per node, edge bytes sorted within a node,
/// child nodes, node values) whose values index the ids and float64 scores.
#[pyclass(frozen)]
struct Seg {
    start: Arr<u32>, ebyte: Arr<u8>, echild: Arr<u32>, val: Arr<u32>, ids: Arr<i32>, scores: Arr<f64>,
    unk_id: i64, unk_score: f64, byte_fallback: bool, byte_ids: [i64; 256],
}

impl Seg {
    #[inline(always)]
    fn child(&self, node: u32, b: u8) -> u32 {
        let st = self.start.s();
        let (lo, hi) = (st[node as usize] as usize, st[node as usize + 1] as usize);
        let s = &self.ebyte.s()[lo..hi];
        if s.len() <= 8 {
            for (i, &x) in s.iter().enumerate() { if x == b { return self.echild.s()[lo + i]; } }
            NONE
        } else {
            match s.binary_search(&b) { Ok(i) => self.echild.s()[lo + i], Err(_) => NONE }
        }
    }

    fn get(&self, key: &[u8]) -> u32 {
        let mut node = 0u32;
        for &b in key { node = self.child(node, b); if node == NONE { return NONE; } }
        self.val.s()[node as usize]
    }

    /// The ids of an unknown run, as laghu's `_unk`: a piece if it spells one, else its bytes under byte fallback, else the unknown id.
    fn unk(&self, p: &str) -> Result<Vec<i64>, String> {
        let k = self.get(p.as_bytes());
        if k != NONE { return Ok(vec![self.ids.s()[k as usize] as i64]); }
        if self.byte_fallback {
            let v: Option<Vec<i64>> = p.bytes().map(|x| { let q = self.byte_ids[x as usize]; if q >= 0 { Some(q) } else { None } }).collect();
            if let Some(v) = v { return Ok(v); }
        }
        if self.unk_id < 0 { return Err(format!("Encountered an unknown token but `unk_id` is missing: {:?}", p)); }
        Ok(vec![self.unk_id])
    }

    /// laghu's `viterbi_trie` and `_path` on byte offsets: candidates by start ascending, then length ascending, kept on a strict `>`.
    fn viterbi(&self, s: &str, out: &mut Vec<i64>) -> Result<(), String> {
        let b = s.as_bytes();
        let n = b.len();
        if n == 0 { return Ok(()); }
        let (ids, scores, val) = (self.ids.s(), self.scores.s(), self.val.s());
        let mut best = vec![0f64; n + 1];
        let mut set = vec![false; n + 1];
        let mut start = vec![0usize; n + 1];
        let mut tid = vec![0i64; n + 1];
        set[0] = true;
        let mut i = 0;
        while i < n {
            let cl = utf8_len(b[i]);
            let bi = best[i];
            let mut single = false;
            let (mut node, mut j) = (0u32, i);
            while j < n {
                node = self.child(node, b[j]);
                if node == NONE { break; }
                j += 1;
                let v = val[node as usize];
                if v != NONE {
                    let c = scores[v as usize] + bi;
                    if !set[j] || c > best[j] { best[j] = c; set[j] = true; start[j] = i; tid[j] = ids[v as usize] as i64; }
                    if j == i + cl { single = true; }
                }
            }
            if !single {
                let j = i + cl;
                let c = self.unk_score + bi;
                if !set[j] || c > best[j] { best[j] = c; set[j] = true; start[j] = i; tid[j] = self.unk_id; }
            }
            i += cl;
        }
        let mut rev: Vec<i64> = Vec::new();
        let (mut e, mut run) = (n, 0usize);
        while e > 0 {
            if tid[e] == self.unk_id { if run == 0 { run = e; } }
            else {
                if run != 0 { let u = self.unk(&s[e..run])?; rev.extend(u.iter().rev()); run = 0; }
                rev.push(tid[e]);
            }
            e = start[e];
            if run != 0 && e == 0 { let u = self.unk(&s[..run])?; rev.extend(u.iter().rev()); }
        }
        out.extend(rev.iter().rev());
        Ok(())
    }

    /// The ids of text `i` of `texts`, id `drop` removed and cut to `max_len`.
    fn text(&self, texts: &Texts, i: usize, drop: Option<i64>, max_len: Option<usize>) -> Result<Vec<i64>, String> {
        let mut out = Vec::new();
        match texts {
            Texts::Split(v) => for it in &v[i] { match it { Item::W(s) => self.viterbi(s, &mut out)?, Item::I(k) => out.push(*k) } },
            Texts::Whole(v, pre) => pre.each(&v[i], |p| match p { Part::W(w) => self.viterbi(w, &mut out), Part::I(k) => { out.push(k); Ok(()) } })?,
        }
        Ok(finish(out, drop, max_len))
    }
}

/// `texts` as whole strings when `pre` is given, else as lists of pre-tokens (str) and added-token ids (int).
fn extract<'a>(texts: &Bound<'_, PyList>, pre: Option<&'a Pre>) -> PyResult<Texts<'a>> {
    if let Some(p) = pre { return Ok(Texts::Whole(texts.extract()?, p)); }
    let mut all = Vec::with_capacity(texts.len());
    for t in texts.iter() {
        let t = t.cast_into::<PyList>()?;
        let mut v = Vec::with_capacity(t.len());
        for x in t.iter() {
            if x.is_instance_of::<PyString>() { v.push(Item::W(x.extract::<String>()?)); } else { v.push(Item::I(x.extract::<i64>()?)); }
        }
        all.push(v);
    }
    Ok(Texts::Split(all))
}

/// Map `f` over `n` indices on `threads` threads, results in index order.
fn par_map<R: Send>(n: usize, threads: usize, f: impl Fn(usize) -> R + Sync) -> Vec<R> {
    if threads <= 1 || n < 2 { return (0..n).map(f).collect(); }
    let next = AtomicUsize::new(0);
    let mut parts: Vec<Vec<(usize, R)>> = Vec::new();
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads.min(n)).map(|_| sc.spawn(|| {
            let mut got = Vec::new();
            loop {
                let a = next.fetch_add(8, Ordering::Relaxed);
                if a >= n { break; }
                for i in a..(a + 8).min(n) { got.push((i, f(i))); }
            }
            got
        })).collect();
        for h in hs { parts.push(h.join().unwrap()); }
    });
    let mut out: Vec<Option<R>> = (0..n).map(|_| None).collect();
    for p in parts { for (i, r) in p { out[i] = Some(r); } }
    out.into_iter().map(|o| o.unwrap()).collect()
}

/// Ids `out` with id `drop` removed and cut to `max_len`, as model2vec's `tokenize` does after tokenizing.
fn finish(mut out: Vec<i64>, drop: Option<i64>, max_len: Option<usize>) -> Vec<i64> {
    if let Some(d) = drop { out.retain(|&k| k != d); }
    if let Some(m) = max_len { out.truncate(m); }
    out
}

/// The mean of the rows of float32 `table` (rows x dim, C order) at the ids `ids(i)` of each of `n` texts, written into float32 `out`
/// (n x dim), as numpy's `mean(axis=0)`: the first row copied, the rest added in order, divided by the count. Returns the counts.
fn pool(py: Python<'_>, n: usize, threads: usize, table: &Bound<'_, PyAny>, out: &Bound<'_, PyAny>,
        ids: impl Fn(usize) -> Result<Vec<i64>, String> + Sync) -> PyResult<Vec<usize>> {
    let tb = PyBuffer::<f32>::get(table)?;
    let ob = PyBuffer::<f32>::get(out)?;
    if tb.dimensions() != 2 || !tb.is_c_contiguous() { return Err(PyValueError::new_err("table must be 2-D C-contiguous float32")); }
    let (rows, dim) = (tb.shape()[0], tb.shape()[1]);
    if ob.readonly() || !ob.is_c_contiguous() || ob.item_count() != n * dim { return Err(PyValueError::new_err("out must be writable float32 (texts, dim)")); }
    let (tp, op) = (tb.buf_ptr() as usize, ob.buf_ptr() as usize);
    let r = py.detach(|| par_map(n, threads, |i| -> Result<usize, String> {
        let ids = ids(i)?;
        let table = unsafe { std::slice::from_raw_parts(tp as *const f32, rows * dim) };
        let o = unsafe { std::slice::from_raw_parts_mut((op as *mut f32).add(i * dim), dim) };
        if ids.is_empty() { o.fill(0.0); return Ok(0); }
        for &k in &ids { if k < 0 || k as usize >= rows { return Err(format!("id {} out of range", k)); } }
        o.copy_from_slice(&table[ids[0] as usize * dim..(ids[0] as usize + 1) * dim]);
        for &k in &ids[1..] {
            let row = &table[k as usize * dim..(k as usize + 1) * dim];
            for d in 0..dim { o[d] += row[d]; }
        }
        let nf = ids.len() as f32;
        for d in 0..dim { o[d] /= nf; }
        Ok(ids.len())
    }));
    r.into_iter().collect::<Result<Vec<_>, _>>().map_err(PyValueError::new_err)
}

#[pymethods]
impl Seg {
    /// Segment with the byte trie in buffers `start` (u32), `ebyte` (u8), `echild` (u32), `val` (u32) over `ids` (i32) and `scores` (f64).
    #[new]
    #[pyo3(signature = (start, ebyte, echild, val, ids, scores, unk_id, unk_score, byte_fallback))]
    fn new(
        start: &Bound<'_, PyAny>, ebyte: &Bound<'_, PyAny>, echild: &Bound<'_, PyAny>, val: &Bound<'_, PyAny>,
        ids: &Bound<'_, PyAny>, scores: &Bound<'_, PyAny>, unk_id: i64, unk_score: f64, byte_fallback: bool,
    ) -> PyResult<Seg> {
        let mut s = Seg {
            start: Arr::get(start, "start")?, ebyte: Arr::get(ebyte, "ebyte")?, echild: Arr::get(echild, "echild")?,
            val: Arr::get(val, "val")?, ids: Arr::get(ids, "ids")?, scores: Arr::get(scores, "scores")?,
            unk_id, unk_score, byte_fallback, byte_ids: [-1; 256],
        };
        let (nodes, edges, nk) = (s.val.len, s.ebyte.len, s.ids.len);
        let bad = |m: &str| Err(PyValueError::new_err(format!("malformed byte trie: {m}")));
        if nodes == 0 || s.start.len != nodes + 1 || s.echild.len != edges || s.scores.len != nk { return bad("array lengths"); }
        let st = s.start.s();
        if st[0] != 0 || st[nodes] as usize != edges || st.windows(2).any(|w| w[0] > w[1]) { return bad("edge starts"); }
        if s.echild.s().iter().any(|&c| c == 0 || c as usize >= nodes) { return bad("child nodes"); }
        if s.val.s().iter().any(|&v| v != NONE && v as usize >= nk) { return bad("node values"); }
        for b in 0..256usize {
            let k = s.get(format!("<0x{:02X}>", b).as_bytes());
            if k != NONE { s.byte_ids[b] = s.ids.s()[k as usize] as i64; }
        }
        Ok(s)
    }

    /// The ids of one pre-token, as laghu's `viterbi_trie`.
    fn word(&self, py: Python<'_>, s: &str) -> PyResult<Vec<i64>> {
        let mut out = Vec::new();
        py.detach(|| self.viterbi(s, &mut out)).map_err(PyValueError::new_err)?;
        Ok(out)
    }

    /// The ids of each text, id `drop` removed and cut to `max_len`. A text is a str split by `pre` when it is given,
    /// else a list of pre-tokens (str) and added-token ids (int).
    #[pyo3(signature = (texts, drop=None, max_len=None, threads=1, pre=None))]
    fn batch(&self, py: Python<'_>, texts: &Bound<'_, PyList>, drop: Option<i64>, max_len: Option<usize>, threads: usize,
             pre: Option<Bound<'_, Pre>>) -> PyResult<Vec<Vec<i64>>> {
        let items = extract(texts, pre.as_ref().map(|p| p.get()))?;
        let r = py.detach(|| par_map(items.len(), threads, |i| self.text(&items, i, drop, max_len)));
        r.into_iter().collect::<Result<Vec<_>, _>>().map_err(PyValueError::new_err)
    }

    /// `batch`, then the mean of each text's rows of float32 `table` (rows x dim, C order) written into float32 `out`
    /// (texts x dim), as `pool` computes it. Returns the counts.
    #[pyo3(signature = (texts, table, out, drop=None, max_len=None, threads=1, pre=None))]
    fn encode(&self, py: Python<'_>, texts: &Bound<'_, PyList>, table: &Bound<'_, PyAny>, out: &Bound<'_, PyAny>,
              drop: Option<i64>, max_len: Option<usize>, threads: usize, pre: Option<Bound<'_, Pre>>) -> PyResult<Vec<usize>> {
        let items = extract(texts, pre.as_ref().map(|p| p.get()))?;
        pool(py, items.len(), threads, table, out, |i| self.text(&items, i, drop, max_len))
    }
}

/// `tokenizers`' BertNormalizer `is_whitespace`: tab, newline and carriage return, and Rust's `char::is_whitespace`.
fn bert_white(c: char) -> bool { matches!(c, '\t' | '\n' | '\r') || c.is_whitespace() }

/// `tokenizers`' BertNormalizer `is_control`: Unicode's Other categories, except tab, newline and carriage return.
fn bert_control(c: char) -> bool { !matches!(c, '\t' | '\n' | '\r') && c.is_other() }

/// `tokenizers`' BertNormalizer `is_chinese_char`: the CJK Unified Ideographs blocks and their extensions and compatibility blocks.
fn bert_chinese(c: char) -> bool {
    matches!(c as u32, 0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF | 0x2A700..=0x2B73F | 0x2B740..=0x2B81F
        | 0x2B920..=0x2CEAF | 0xF900..=0xFAFF | 0x2F800..=0x2FA1F)
}

/// The words of normalized text `s`, as `tokenizers`' BertPreTokenizer splits it: whitespace removed, each punctuation mark a word.
fn bert_words(s: &str) -> Vec<&str> {
    let mut out = Vec::with_capacity(s.len() / 4 + 1);
    let mut start = None;
    for (i, c) in s.char_indices() {
        if c.is_whitespace() {
            if let Some(a) = start.take() { out.push(&s[a..i]); }
        } else if c.is_ascii_punctuation() || c.is_punctuation() {
            if let Some(a) = start.take() { out.push(&s[a..i]); }
            out.push(&s[i..i + c.len_utf8()]);
        } else if start.is_none() { start = Some(i); }
    }
    if let Some(a) = start { out.push(&s[a..]); }
    out
}

/// The ids of a tokenizer.json with a BertNormalizer, a BertPreTokenizer and a WordPiece model, as `encode_single_sequence` gives
/// them, without the offsets and alignments of the `tokenizers` crate. Ported from `FastWordPiece` in model2vec-rs, copyright (c) 2025
/// The Minish Lab, MIT licence (rust/LICENSE-model2vec-rs). Unlike it, pieces are looked up in the model's vocabulary alone, as
/// `WordPiece::tokenize` looks them up, and NFD and the character classes come from the crates the `tokenizers` crate uses.
#[pyclass(frozen)]
struct WordPiece {
    vocab: HashMap<String, u32>, single: [i64; 128], unk: i64, max_chars: usize, prefix: String,
    clean: bool, chinese: bool, strip: bool, lower: bool,
    added: Vec<(String, i64)>, first: [bool; 256],
}

impl WordPiece {
    /// The ids of text `s`: added tokens matched in the raw text, leftmost and longest first, and the text between them normalized,
    /// split into words and segmented, as `AddedVocabulary::extract_and_normalize` and the pipeline do.
    fn ids(&self, s: &str) -> Vec<i64> {
        let (b, mut out, mut buf) = (s.as_bytes(), Vec::new(), String::new());
        let (mut start, mut i) = (0, 0);
        while i < b.len() {
            let hit = if self.first[b[i] as usize] { self.added.iter().find(|(c, _)| s[i..].starts_with(c.as_str())) } else { None };
            match hit {
                Some((c, k)) => { self.span(&s[start..i], &mut out, &mut buf); out.push(*k); i += c.len(); start = i; }
                None => i += utf8_len(b[i]),
            }
        }
        self.span(&s[start..], &mut out, &mut buf);
        out
    }

    fn span(&self, s: &str, out: &mut Vec<i64>, buf: &mut String) {
        if s.is_empty() { return; }
        let n = self.normalize(s);
        for w in bert_words(&n) { self.word(w, out, buf); }
    }

    /// Text `s` as `tokenizers`' BertNormalizer leaves it: cleaned, CJK ideographs spaced, accents stripped, lowercased.
    fn normalize(&self, s: &str) -> String {
        if s.is_ascii() {
            let mut b = Vec::with_capacity(s.len());
            for &x in s.as_bytes() {
                let x = match x { b'\t' | b'\n' | b'\r' if self.clean => b' ', 0..=0x1f | 0x7f if self.clean => continue, _ => x };
                b.push(if self.lower { x.to_ascii_lowercase() } else { x });
            }
            return String::from_utf8(b).unwrap();
        }
        let mut t = String::with_capacity(s.len());
        for c in s.chars() {
            if self.clean {
                if c == '\0' || c == '\u{fffd}' || bert_control(c) { continue; }
                if bert_white(c) { t.push(' '); continue; }
            }
            if self.chinese && bert_chinese(c) { t.push(' '); t.push(c); t.push(' '); } else { t.push(c); }
        }
        if self.strip {
            let mut u = String::with_capacity(t.len());
            for (c, _) in t.nfd() {
                if c.is_mark_nonspacing() { continue; }
                if self.lower { u.extend(c.to_lowercase()); } else { u.push(c); }
            }
            return u;
        }
        if self.lower { t.chars().flat_map(char::to_lowercase).collect() } else { t }
    }

    /// The ids of word `w` by `WordPiece::tokenize`'s greedy longest match, or the unknown id when the word is too long or a part has
    /// no piece.
    fn word(&self, w: &str, out: &mut Vec<i64>, buf: &mut String) {
        if w.len() > self.max_chars && w.chars().count() > self.max_chars { out.push(self.unk); return; }
        if w.len() == 1 { let k = self.single[w.as_bytes()[0] as usize]; out.push(if k >= 0 { k } else { self.unk }); return; }
        let (mark, mut start) = (out.len(), 0);
        while start < w.len() {
            let (mut end, mut found) = (w.len(), None);
            while end > start {
                let sub = &w[start..end];
                let k = if start == 0 { self.vocab.get(sub) } else {
                    buf.clear(); buf.push_str(&self.prefix); buf.push_str(sub); self.vocab.get(buf.as_str())
                };
                if let Some(&k) = k { found = Some(k); break; }
                end -= sub.chars().next_back().map_or(1, char::len_utf8);
            }
            match found {
                Some(k) => { out.push(k as i64); start = end; }
                None => { out.truncate(mark); out.push(self.unk); return; }
            }
        }
    }

    fn text(&self, s: &str, drop: Option<i64>, max_len: Option<usize>) -> Vec<i64> { finish(self.ids(s), drop, max_len) }
}

#[pymethods]
impl WordPiece {
    /// Build from tokenizer JSON `json`. Raises `ValueError` for another model, normalizer or pre-tokenizer, a vocabulary without
    /// its unknown token, or an added token matched after normalization or with `lstrip`, `rstrip` or `single_word`.
    #[new]
    fn new(json: &str) -> PyResult<WordPiece> {
        let bad = |m: String| Err(PyValueError::new_err(m));
        let tok = Tokenizer::from_str(json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ModelWrapper::WordPiece(m) = tok.get_model() else { return bad("model is not WordPiece".into()) };
        let Some(NormalizerWrapper::BertNormalizer(n)) = tok.get_normalizer() else { return bad("normalizer is not BertNormalizer".into()) };
        let Some(PreTokenizerWrapper::BertPreTokenizer(_)) = tok.get_pre_tokenizer() else { return bad("pre-tokenizer is not BertPreTokenizer".into()) };
        let vocab = m.get_vocab();
        let Some(&unk) = vocab.get(&m.unk_token) else { return bad(format!("unknown token {:?} not in the vocabulary", m.unk_token)) };
        let mut added = Vec::new();
        for (k, t) in tok.get_added_tokens_decoder() {
            if t.content.is_empty() || t.normalized || t.lstrip || t.rstrip || t.single_word {
                return bad(format!("added token {:?} is empty, normalized, or lstrip, rstrip or single_word", t.content));
            }
            added.push((t.content, k as i64));
        }
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.0.cmp(&b.0)));
        let mut first = [false; 256];
        for (c, _) in &added { first[c.as_bytes()[0] as usize] = true; }
        let mut single = [-1i64; 128];
        for (b, k) in single.iter_mut().enumerate() { if let Some(&v) = vocab.get(&(b as u8 as char).to_string()) { *k = v as i64; } }
        Ok(WordPiece {
            single, unk: unk as i64, max_chars: m.max_input_chars_per_word, prefix: m.continuing_subword_prefix.clone(),
            clean: n.clean_text, chinese: n.handle_chinese_chars, strip: n.strip_accents.unwrap_or(n.lowercase), lower: n.lowercase,
            added, first, vocab,
        })
    }

    /// The ids of each text, id `drop` removed and cut to `max_len`.
    #[pyo3(signature = (texts, drop=None, max_len=None, threads=1))]
    fn batch(&self, py: Python<'_>, texts: Vec<String>, drop: Option<i64>, max_len: Option<usize>, threads: usize) -> Vec<Vec<i64>> {
        py.detach(|| par_map(texts.len(), threads, |i| self.text(&texts[i], drop, max_len)))
    }

    /// `batch`, then the mean of each text's rows of float32 `table` written into float32 `out`, as `pool` computes it. Returns the counts.
    #[pyo3(signature = (texts, table, out, drop=None, max_len=None, threads=1))]
    fn encode(&self, py: Python<'_>, texts: Vec<String>, table: &Bound<'_, PyAny>, out: &Bound<'_, PyAny>,
              drop: Option<i64>, max_len: Option<usize>, threads: usize) -> PyResult<Vec<usize>> {
        pool(py, texts.len(), threads, table, out, |i| Ok(self.text(&texts[i], drop, max_len)))
    }
}

#[pymodule]
fn laghu_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Seg>()?;
    m.add_class::<Pre>()?;
    m.add_class::<WordPiece>()?;
    m.add("TOKENIZERS", TOKENIZERS)?;
    Ok(())
}
