//! The optional Rust core of laghu: Unigram Viterbi over byte offsets, searching a CSR byte trie mapped from laghu's
//! vocabulary file, and mean pooling over a mapped float32 table, batched over threads with the GIL released.
//! Python still normalizes and pre-tokenizes; every result must equal laghu's Python path, and model2vec's, bit for bit.
use pyo3::buffer::{Element, PyBuffer};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
use std::sync::atomic::{AtomicUsize, Ordering};

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

enum Item { W(String), I(i64) }

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

    fn text(&self, items: &[Item], drop: Option<i64>, max_len: Option<usize>) -> Result<Vec<i64>, String> {
        let mut out = Vec::new();
        for it in items { match it { Item::W(s) => self.viterbi(s, &mut out)?, Item::I(k) => out.push(*k) } }
        if let Some(d) = drop { out.retain(|&k| k != d); }
        if let Some(m) = max_len { out.truncate(m); }
        Ok(out)
    }
}

fn extract(texts: &Bound<'_, PyList>) -> PyResult<Vec<Vec<Item>>> {
    let mut all = Vec::with_capacity(texts.len());
    for t in texts.iter() {
        let t = t.cast_into::<PyList>()?;
        let mut v = Vec::with_capacity(t.len());
        for x in t.iter() {
            if x.is_instance_of::<PyString>() { v.push(Item::W(x.extract::<String>()?)); } else { v.push(Item::I(x.extract::<i64>()?)); }
        }
        all.push(v);
    }
    Ok(all)
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

    /// The ids of each text, given as a list of pre-tokens (str) and added-token ids (int), id `drop` removed and cut to `max_len`.
    #[pyo3(signature = (texts, drop=None, max_len=None, threads=1))]
    fn batch(&self, py: Python<'_>, texts: &Bound<'_, PyList>, drop: Option<i64>, max_len: Option<usize>, threads: usize) -> PyResult<Vec<Vec<i64>>> {
        let items = extract(texts)?;
        let r = py.detach(|| par_map(items.len(), threads, |i| self.text(&items[i], drop, max_len)));
        r.into_iter().collect::<Result<Vec<_>, _>>().map_err(PyValueError::new_err)
    }

    /// `batch`, then the mean of each text's rows of float32 `table` (rows x dim, C order) written into float32 `out`
    /// (texts x dim), as numpy's `mean(axis=0)`: the first row copied, the rest added in order, divided by the count. Returns the counts.
    #[pyo3(signature = (texts, table, out, drop=None, max_len=None, threads=1))]
    fn encode(&self, py: Python<'_>, texts: &Bound<'_, PyList>, table: &Bound<'_, PyAny>, out: &Bound<'_, PyAny>,
              drop: Option<i64>, max_len: Option<usize>, threads: usize) -> PyResult<Vec<usize>> {
        let items = extract(texts)?;
        let tb = PyBuffer::<f32>::get(table)?;
        let ob = PyBuffer::<f32>::get(out)?;
        if tb.dimensions() != 2 || !tb.is_c_contiguous() { return Err(PyValueError::new_err("table must be 2-D C-contiguous float32")); }
        let (rows, dim) = (tb.shape()[0], tb.shape()[1]);
        if ob.readonly() || !ob.is_c_contiguous() || ob.item_count() != items.len() * dim { return Err(PyValueError::new_err("out must be writable float32 (texts, dim)")); }
        let (tp, op) = (tb.buf_ptr() as usize, ob.buf_ptr() as usize);
        let r = py.detach(|| par_map(items.len(), threads, |i| -> Result<usize, String> {
            let ids = self.text(&items[i], drop, max_len)?;
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
}

#[pymodule]
fn laghu_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Seg>()?;
    Ok(())
}
