//! Own the native engine and the contexts for a contiguous range of layers.
//! A head batch carries token IDs. A downstream batch carries one residual
//! vector per row. A non-tail stage returns residuals for all rows, and a
//! tail exposes logits for the requested rows.

use publicompute_llama_sys as sys;
use std::collections::{BTreeMap, HashSet};
use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::Mutex;

pub use publicompute_proto::service::Message;

/// Packed slot: group (context) in the high bits, local sequence in the low 16 bits.
pub fn slot(gruppo: u32, locale: u32) -> u32 {
    (gruppo << 16) | locale
}

pub struct EngineOptions {
    pub model: String,
    pub layer_start: i32,
    /// -1 = through the last layer.
    pub layer_end: i32,
    /// GPU layers (0 = CPU, 999 = all).
    pub ngl: i32,
    pub threads: i32,
    /// Number of sequences (slots) in the context.
    pub slots: u32,
    /// Give each group its own context so alternating groups can reuse their compute graphs.
    pub groups: u32,
    pub ctx_per_slot: u32,
    /// Max batch size in tokens (block prefill).
    pub n_batch: u32,
    /// Record the experts picked by the router (`cb_eval` callback on the `ffn_moe_topk` tensor). Adds overhead.
    pub experts: bool,
    /// Keep up to this many recurrent-state snapshots per sequence for draft
    /// rewind. Budget one working state plus that many snapshots per slot.
    pub max_draft_tokens: u32,
    /// Load the tail's draft block and allocate a separate draft context for
    /// each group. The main context exposes hidden states for feeding it.
    pub mtp: bool,
}

/// (rows, k, indices) for one sub-batch.
type Pezzo = (usize, usize, Vec<i32>);

/// Experts captured by the callback: per layer, chunks of `righe × k` indices, one per sub-batch.
#[derive(Default)]
struct Cattura {
    pezzi: Mutex<BTreeMap<i32, Vec<Pezzo>>>,
    nomi_moe: Mutex<HashSet<String>>,
}

/// Experts for a batch row: (row index, layer, experts by score).
pub type EspertiRiga = (usize, i32, Vec<i32>);

/// Request and read `ffn_moe_topk-<layer>` tensors. Each holds k selected
/// expert indices per token as signed 32-bit integers.
/// With `PUBLICOMPUTE_ELENCO_TENSORI` set, print the "moe" tensor names seen, once.
unsafe extern "C" fn cb_eval(t: *mut sys::ggml_tensor, ask: bool, ud: *mut c_void) -> bool {
    let cat = &*(ud as *const Cattura);
    let name = CStr::from_ptr((*t).name.as_ptr()).to_string_lossy();
    if ask && std::env::var_os("PUBLICOMPUTE_ELENCO_TENSORI").is_some() && name.contains("moe") {
        let base = name.rsplit_once('-').map_or(name.as_ref(), |x| x.0).to_string();
        if cat.nomi_moe.lock().unwrap().insert(base.clone()) {
            eprintln!("[experts] tensor {base} type {}", (*t).type_);
        }
    }
    let Some(layer) = name.strip_prefix("ffn_moe_topk-").and_then(|x| x.parse::<i32>().ok()) else {
        return !ask; // Tensor not needed: decline the request, or continue if computation already finished.
    };
    if ask {
        return true;
    }
    let (k, n) = ((*t).ne[0] as usize, (*t).ne[1] as usize);
    if (*t).type_ != sys::ggml_type_GGML_TYPE_I32 || k == 0 || n == 0 {
        return true;
    }
    let mut dati = vec![0i32; k * n];
    for r in 0..n {
        // The tensor is a view of the argsort, so rows may be noncontiguous.
        sys::ggml_backend_tensor_get(t, dati.as_mut_ptr().add(r * k) as *mut c_void, r * (*t).nb[1], k * 4);
    }
    cat.pezzi.lock().unwrap().entry(layer).or_default().push((n, k, dati));
    true
}

pub struct Engine {
    model: *mut sys::llama_model,
    ctxs: Vec<*mut sys::llama_context>,
    /// Context of the last batch (for `logits`).
    last: usize,
    vocab: *const sys::llama_vocab,
    /// Residual width in values per row, including all residual streams.
    /// Advertise this width as n_embd in messages between stages.
    pub n_embd: usize,
    /// Number of residual streams in one row. Per-stream int8 uses this count
    /// to place a separate scale before each segment.
    pub stream_count: usize,
    pub n_vocab: usize,
    pub n_layer: i32,
    pub layer_start: i32,
    pub layer_end: i32,
    pub head: bool,
    pub tail: bool,
    pub n_batch: usize,
    /// Slots (sequences) per context.
    pub per_ctx: u32,
    /// One draft context per group when the tail loads the draft block.
    ctxs_mtp: Vec<*mut sys::llama_context>,
    /// MTP context of the last MTP batch (for `mtp_logits`).
    ultimo_mtp: usize,
    /// Width of the hidden state the MTP block takes with each token (`llama_model_n_embd_out`); 0 without MTP.
    pub mtp_state_width: usize,
    /// Retain the draft block's output state when another proposal may follow.
    /// A single proposal needs only the logits.
    mtp_stato_uscita: bool,
    cattura: Option<Box<Cattura>>,
    /// Batches with a mismatched captured row count, which were not logged.
    pub esperti_persi: usize,
    esperti_ult: Vec<EspertiRiga>,
}

/// A draft row pairs the token at pos with the main network's hidden state
/// from pos - 1. The sequence packs the group and the local slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MtpRow {
    pub token: i32,
    pub pos: i32,
    pub seq: i32,
    /// Ask for the logits of this row (the draft of the token after `token`).
    pub logits: bool,
}

/// One batch row.
#[derive(Clone, Copy)]
/// `seq` is a packed slot (see [`slot`]).
pub struct Row {
    pub token: i32,
    pub pos: i32,
    pub seq: i32,
    /// Request logits for this row at the tail only. Other stages always return state for all rows.
    pub logits: bool,
}

// Use the engine from one thread at a time. Each stage has one compute thread.
unsafe impl Send for Engine {}

fn init_backend() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        // Keep logs quiet except for errors.
        sys::llama_log_set(Some(log_silenzioso), std::ptr::null_mut());
        sys::llama_backend_init();
    });
}

static LOG_COMPLETO: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Set before loading the engine. A true value also sends native buffer
/// and memory diagnostics to stderr.
pub fn set_llama_logging(si: bool) {
    LOG_COMPLETO.store(si, std::sync::atomic::Ordering::Relaxed);
}

unsafe extern "C" fn log_silenzioso(livello: u32, text: *const c_char, _: *mut std::ffi::c_void) {
    if (livello >= 4 || LOG_COMPLETO.load(std::sync::atomic::Ordering::Relaxed)) && !text.is_null() {
        eprint!("{}", std::ffi::CStr::from_ptr(text).to_string_lossy());
    }
}

impl Engine {
    /// Load the requested layer range and allocate a context for each group.
    /// Return an error if loading fails, a requested draft block is unavailable,
    /// or a context cannot be allocated.
    pub fn load(o: &EngineOptions) -> Result<Engine, String> {
        init_backend();
        let path = CString::new(o.model.clone()).map_err(|e| e.to_string())?;
        unsafe {
            let mut mp = sys::llama_model_default_params();
            mp.n_gpu_layers = o.ngl;
            mp.layer_start = o.layer_start;
            mp.layer_end = o.layer_end;
            mp.load_mtp = o.mtp;
            let model = sys::llama_model_load_from_file(path.as_ptr(), mp);
            if model.is_null() {
                return Err(format!(
                    "cannot load {} (layer [{}, {})){}",
                    o.model,
                    o.layer_start,
                    o.layer_end,
                    if o.mtp { " with MTP (--mtp): the tail requires a GGUF containing the draft block" } else { "" }
                ));
            }
            let vocab = sys::llama_model_get_vocab(model);
            let n_layer = sys::llama_model_n_layer(model);
            let layer_end = if o.layer_end < 0 { n_layer } else { o.layer_end };
            let mut m = Engine {
                model,
                ctxs: Vec::new(),
                last: 0,
                vocab,
                n_embd: sys::llama_model_n_embd_stage(model) as usize,
                stream_count: (sys::llama_model_n_embd_stage(model) as usize / (sys::llama_model_n_embd(model) as usize).max(1)).max(1),
                n_vocab: sys::llama_vocab_n_tokens(vocab) as usize,
                n_layer,
                layer_start: o.layer_start,
                layer_end,
                head: o.layer_start == 0,
                tail: layer_end == n_layer,
                n_batch: o.n_batch as usize,
                per_ctx: 1,
                ctxs_mtp: Vec::new(),
                ultimo_mtp: 0,
                mtp_state_width: 0,
                mtp_stato_uscita: false,
                cattura: if o.experts { Some(Box::default()) } else { None },
                esperti_persi: 0,
                esperti_ult: Vec::new(),
            };
            if o.mtp {
                // Only create draft contexts if the native loader could load the block.
                if sys::llama_model_n_layer_nextn(model) <= 0 || !m.tail {
                    return Err(format!("{}: --mtp requires the tail of a model with MTP weights", o.model));
                }
                m.mtp_state_width = sys::llama_model_n_embd_out(model) as usize;
            }
            let groups = o.groups.max(1);
            let per_ctx = o.slots.div_ceil(groups).max(1);
            for _ in 0..groups {
                let mut cp = sys::llama_context_default_params();
                cp.n_ctx = o.ctx_per_slot * per_ctx;
                cp.n_batch = o.n_batch;
                cp.n_ubatch = o.n_batch;
                cp.n_seq_max = per_ctx;
                cp.n_threads = o.threads;
                cp.n_threads_batch = o.threads;
                cp.kv_unified = false;
                cp.n_rs_seq = o.max_draft_tokens;
                if let Some(c) = &m.cattura {
                    cp.cb_eval = Some(cb_eval);
                    cp.cb_eval_user_data = &**c as *const Cattura as *mut c_void;
                }
                let ctx = sys::llama_init_from_model(model, cp);
                if ctx.is_null() {
                    return Err("llama_init_from_model failed (possibly out of memory)".into());
                }
                m.ctxs.push(ctx);
                if o.mtp {
                    // Prompt rows also feed the draft cache, so keep their hidden states even
                    // when they do not request logits.
                    sys::llama_stage_set_embeddings_nextn(ctx, true, false);
                    let mut cm = sys::llama_context_default_params();
                    cm.n_ctx = o.ctx_per_slot * per_ctx;
                    // The last prefill batch may feed the first sampled token as one extra row.
                    cm.n_batch = o.n_batch + 1;
                    cm.n_ubatch = o.n_batch + 1;
                    cm.n_seq_max = per_ctx;
                    cm.n_threads = o.threads;
                    cm.n_threads_batch = o.threads;
                    cm.kv_unified = false;
                    cm.ctx_type = sys::llama_context_type_LLAMA_CONTEXT_TYPE_MTP;
                    let c = sys::llama_init_from_model(model, cm);
                    if c.is_null() {
                        return Err("could not create MTP context (llama_init_from_model failed)".into());
                    }
                    if o.max_draft_tokens > 1 {
                        // Expose hidden state for requested rows using the logits index.
                        sys::llama_stage_set_embeddings_nextn(c, true, true);
                        m.mtp_stato_uscita = true;
                    }
                    m.ctxs_mtp.push(c);
                }
            }
            m.per_ctx = per_ctx;
            Ok(m)
        }
    }

    /// Clear the slot in its context memory. The packed slot has a group in the high bits and a local sequence
    /// in the low 16 (see [`slot`]).
    pub fn svuota_posto(&self, slot: u32) {
        unsafe {
            let (g, local) = ((slot >> 16) as usize, (slot & 0xffff) as i32);
            if let Some(&c) = self.ctxs.get(g) {
                let mem = sys::llama_get_memory(c);
                sys::llama_memory_seq_rm(mem, local, -1, -1);
            }
        }
    }

    /// Discard the speculative suffix starting at pos. Recurrent rewind uses
    /// a saved snapshot once, so run a batch before trying another rewind.
    /// Return a conversation error if the engine cannot roll back.
    pub fn riavvolgi(&self, slot: u32, pos: u32) -> Result<(), String> {
        unsafe {
            let (g, local) = ((slot >> 16) as usize, (slot & 0xffff) as i32);
            let Some(&c) = self.ctxs.get(g) else { return Err(format!("slot {slot:#x} is outside the configured range")) };
            let mem = sys::llama_get_memory(c);
            if sys::llama_memory_seq_rm(mem, local, pos as i32, -1) {
                Ok(())
            } else {
                Err(format!("rewind rejected (slot {slot:#x}, from position {pos}): n_rs_seq is too small or the snapshot was already consumed"))
            }
        }
    }

    pub fn ha_mtp(&self) -> bool {
        !self.ctxs_mtp.is_empty()
    }

    /// Borrow the hidden state of a row until the next main batch. The draft
    /// block takes it together with the token immediately after that row.
    pub fn mtp_state(&self, i: usize) -> Result<&[f32], String> {
        if !self.ha_mtp() {
            return Err("MTP block is not loaded".into());
        }
        unsafe {
            let h = sys::llama_stage_get_embeddings_nextn_ith(self.ctxs[self.last], i as i32);
            if h.is_null() {
                return Err(format!("missing MTP state for row {i}"));
            }
            Ok(std::slice::from_raw_parts(h, self.mtp_state_width))
        }
    }

    /// Feed consecutive rows to the cached draft context. Each state row is
    /// paired with the input token that follows it. The caller must rewind any
    /// overlapping cached positions before this call. Rows marked for logits
    /// expose the distribution for the following token.
    pub fn run_mtp(&mut self, rows: &[MtpRow], stati: &[f32]) -> Result<(), String> {
        let n = rows.len();
        if !self.ha_mtp() {
            return Err("MTP block is not loaded".into());
        }
        if n == 0 {
            return Err("empty MTP batch".into());
        }
        let g = (rows[0].seq >> 16) as usize;
        if g >= self.ctxs_mtp.len() || rows.iter().any(|r| (r.seq >> 16) as usize != g) {
            return Err("all MTP batch rows must share one group and context".into());
        }
        if n > self.n_batch + 1 {
            return Err(format!("MTP batch of {n} rows exceeds n_batch + 1 = {}", self.n_batch + 1));
        }
        if stati.len() != n * self.mtp_state_width {
            return Err("wrong state width in MTP batch".into());
        }
        let ctx = self.ctxs_mtp[g];
        self.ultimo_mtp = g;
        unsafe {
            let b = sys::llama_batch_ext_init(ctx);
            if b.is_null() {
                return Err("llama_batch_ext_init failed".into());
            }
            let mut esito = Ok(());
            for (i, r) in rows.iter().enumerate() {
                let idx = sys::llama_batch_ext_add_token(b, r.seq & 0xffff, r.token);
                if idx != i as i32 {
                    esito = Err(format!("llama_batch_ext_add_token returned {idx} for row {i}"));
                    break;
                }
                let pos = r.pos;
                let state = sys::llama_embd { data: stati.as_ptr().add(i * self.mtp_state_width), n_rows: 1, n_embd: self.mtp_state_width };
                if !sys::llama_batch_ext_set_pos(b, idx, &pos)
                    || !sys::llama_batch_ext_set_embd_state(b, idx, state)
                    || !sys::llama_batch_ext_set_output_logits(b, idx, r.logits)
                {
                    esito = Err(format!("engine rejected MTP batch row {i}"));
                    break;
                }
            }
            if esito.is_ok() {
                let t_dec = std::time::Instant::now();
                let rc = sys::llama_process(ctx, sys::llama_process_type_LLAMA_PROCESS_TYPE_DECODE, b);
                crate::timings::aggiungi_mtp(t_dec.elapsed());
                if rc != 0 {
                    esito = Err(format!("llama_process (MTP) ha restituito {rc}"));
                }
            }
            sys::llama_batch_ext_free(b);
            esito
        }
    }

    /// Logits of row `i` of the last MTP batch (only rows with `logits`). Valid until the next MTP batch.
    pub fn mtp_logits(&self, i: usize) -> Result<&[f32], String> {
        if !self.ha_mtp() {
            return Err("MTP block is not loaded".into());
        }
        unsafe {
            let l = sys::llama_get_logits_ith(self.ctxs_mtp[self.ultimo_mtp], i as i32);
            if l.is_null() {
                return Err(format!("missing MTP logits for row {i}"));
            }
            Ok(std::slice::from_raw_parts(l, self.n_vocab))
        }
    }

    /// Borrow draft output state until the next draft batch. Pair it with its
    /// proposed token to extend the draft by another step.
    pub fn mtp_stato_uscita(&self, i: usize) -> Result<&[f32], String> {
        if !self.mtp_stato_uscita {
            return Err("MTP output state requires --draft-max-tokens > 1 on the tail".into());
        }
        unsafe {
            let h = sys::llama_stage_get_embeddings_nextn_ith(self.ctxs_mtp[self.ultimo_mtp], i as i32);
            if h.is_null() {
                return Err(format!("missing MTP output state for row {i}"));
            }
            Ok(std::slice::from_raw_parts(h, self.mtp_state_width))
        }
    }

    /// Drop draft context positions from pos onward for the slot. With no
    /// draft context this is a no-op.
    pub fn mtp_riavvolgi(&self, slot: u32, pos: u32) -> Result<(), String> {
        unsafe {
            let (g, local) = ((slot >> 16) as usize, (slot & 0xffff) as i32);
            let Some(&c) = self.ctxs_mtp.get(g) else { return Ok(()) };
            if sys::llama_memory_seq_rm(sys::llama_get_memory(c), local, pos as i32, -1) {
                Ok(())
            } else {
                Err(format!("MTP cache rewind rejected (slot {slot:#x}, from position {pos})"))
            }
        }
    }

    /// Empties the place in the MTP cache (nothing to do without MTP).
    pub fn mtp_svuota_posto(&self, slot: u32) {
        unsafe {
            let (g, local) = ((slot >> 16) as usize, (slot & 0xffff) as i32);
            if let Some(&c) = self.ctxs_mtp.get(g) {
                sys::llama_memory_seq_rm(sys::llama_get_memory(c), local, -1, -1);
            }
        }
    }

    pub fn groups(&self) -> usize {
        self.ctxs.len()
    }

    /// Is this slot valid for the current configuration?
    pub fn posto_valido(&self, slot: u32) -> bool {
        ((slot >> 16) as usize) < self.ctxs.len() && (slot & 0xffff) < self.per_ctx
    }

    /// Run one batch. `state` is the input (rows × state width) for stages downstream of the head.
    /// Return output state for every row before the tail, or `None` at the tail, where requested row
    /// logits can be read with [`Motore::logits`].
    pub fn run(&mut self, rows: &[Row], state: Option<&[f32]>) -> Result<Option<Vec<f32>>, String> {
        let n = rows.len();
        if n == 0 {
            return Err("empty batch".into());
        }
        let g = (rows[0].seq >> 16) as usize;
        if g >= self.ctxs.len() || rows.iter().any(|r| (r.seq >> 16) as usize != g) {
            return Err("all batch rows must share one group and context".into());
        }
        let ctx = self.ctxs[g];
        self.last = g;
        if n > self.n_batch {
            return Err(format!("batch of {n} rows exceeds n_batch = {}", self.n_batch));
        }
        if !self.head && state.map_or(true, |s| s.len() != n * self.n_embd) {
            return Err("missing input state or wrong state width".into());
        }
        unsafe {
            let mut b = sys::llama_batch_init(n as i32, if self.head { 0 } else { self.n_embd as i32 }, 1);
            b.n_tokens = n as i32;
            for (i, r) in rows.iter().enumerate() {
                if self.head {
                    *b.token.add(i) = r.token;
                } else {
                    let s = state.unwrap();
                    std::ptr::copy_nonoverlapping(s.as_ptr().add(i * self.n_embd), b.embd.add(i * self.n_embd), self.n_embd);
                }
                *b.pos.add(i) = r.pos;
                *b.n_seq_id.add(i) = 1;
                *(*b.seq_id.add(i)) = r.seq & 0xffff;
                *b.logits.add(i) = if self.tail { r.logits as i8 } else { 1 };
            }
            if let Some(c) = &self.cattura {
                c.pezzi.lock().unwrap().clear();
            }
            let t_dec = std::time::Instant::now();
            let rc = sys::llama_decode(ctx, b);
            crate::timings::aggiungi(crate::timings::Fase::Decode, t_dec.elapsed());
            sys::llama_batch_free(b);
            if rc != 0 {
                return Err(format!("llama_decode returned {rc} (stage [{}, {}))", self.layer_start, self.layer_end));
            }
            if self.cattura.is_some() {
                self.raccogli_esperti(rows);
            }
            if self.tail {
                return Ok(None);
            }
            let mut out = vec![0f32; n * self.n_embd];
            let _t = crate::timings::Cronometro::new(crate::timings::Fase::Sincro);
            for i in 0..n {
                let e = sys::llama_get_embeddings_ith(ctx, i as i32);
                if e.is_null() {
                    return Err(format!("missing state for row {i}"));
                }
                std::ptr::copy_nonoverlapping(e, out.as_mut_ptr().add(i * self.n_embd), self.n_embd);
            }
            Ok(Some(out))
        }
    }

    /// Put experts captured in the last batch back in row order. Applies to generation batches (one
    /// row per distinct sequence). Prefill batches record nothing. See `esperti::ordine_sotto_lotti`.
    fn raccogli_esperti(&mut self, rows: &[Row]) {
        self.esperti_ult.clear();
        let seq: Vec<i32> = rows.iter().map(|r| r.seq & 0xffff).collect();
        let mut unici = seq.clone();
        unici.sort_unstable();
        unici.dedup();
        let Some(c) = &self.cattura else { return };
        let pezzi = std::mem::take(&mut *c.pezzi.lock().unwrap());
        if unici.len() != seq.len() {
            return;
        }
        let ordine = crate::experts::ordine_sotto_lotti(&seq);
        let mut persi = false;
        for (layer, v) in pezzi {
            let mut per_riga: Vec<Vec<i32>> = Vec::with_capacity(seq.len());
            for (n, k, dati) in v {
                for r in 0..n {
                    per_riga.push(dati[r * k..(r + 1) * k].to_vec());
                }
            }
            if per_riga.len() != seq.len() {
                persi = true;
                continue;
            }
            for (pos, e) in per_riga.into_iter().enumerate() {
                self.esperti_ult.push((ordine[pos], layer, e));
            }
        }
        if persi {
            self.esperti_persi += 1;
        }
    }

    /// Experts from the last batch (empty if not recorded). Reading drains them.
    pub fn prendi_esperti(&mut self) -> Vec<EspertiRiga> {
        std::mem::take(&mut self.esperti_ult)
    }

    /// Logits for row `i` in the last batch (tail only, rows with `logits` only). Valid until the next batch.
    pub fn logits(&self, i: usize) -> Result<&[f32], String> {
        unsafe {
            let t_sync = std::time::Instant::now();
            let l = sys::llama_get_logits_ith(self.ctxs[self.last], i as i32);
            crate::timings::aggiungi(crate::timings::Fase::Sincro, t_sync.elapsed());
            if l.is_null() {
                return Err(format!("missing logits for row {i}"));
            }
            Ok(std::slice::from_raw_parts(l, self.n_vocab))
        }
    }

    /// `general.architecture` of the loaded model ("" if unreadable).
    pub fn architettura(&self) -> String {
        let mut buf = vec![0u8; 128];
        let n = unsafe { sys::llama_model_meta_val_str(self.model, c"general.architecture".as_ptr(), buf.as_mut_ptr() as *mut c_char, buf.len()) };
        if n <= 0 || n as usize >= buf.len() {
            return String::new();
        }
        String::from_utf8_lossy(&buf[..n as usize]).into_owned()
    }

    pub fn e_eog(&self, token: i32) -> bool {
        unsafe { sys::llama_vocab_is_eog(self.vocab, token) }
    }

    /// Apply the model's chat template to a user message (CLI tests) and tokenize.
    pub fn prepara_prompt(&self, utente: &str) -> Result<(Vec<i32>, String, bool), String> {
        self.prepara_conversazione(&[Message { role: "user".into(), text: utente.to_string() }])
    }

    /// Template and tokenize the supplied conversation. Return the token IDs,
    /// rendered prompt, and whether the native template API produced it.
    /// Some templates use a local formatter when the native API cannot render them.
    pub fn prepara_conversazione(&self, msgs: &[Message]) -> Result<(Vec<i32>, String, bool), String> {
        if msgs.is_empty() {
            return Err("empty conversation".into());
        }
        let ruoli: Vec<CString> = msgs.iter().map(|m| CString::new(m.role.as_str()).map_err(|_| "role contains a NUL byte")).collect::<Result<_, _>>()?;
        let contenuti: Vec<CString> = msgs.iter().map(|m| CString::new(m.text.as_str()).map_err(|_| "prompt contains a NUL byte")).collect::<Result<_, _>>()?;
        let c_msgs: Vec<sys::llama_chat_message> = ruoli.iter().zip(&contenuti).map(|(r, c)| sys::llama_chat_message { role: r.as_ptr(), content: c.as_ptr() }).collect();
        let lunghezza: usize = msgs.iter().map(|m| m.text.len() + m.role.len()).sum();
        let (text, usato) = unsafe {
            let tmpl = sys::llama_model_chat_template(self.model, std::ptr::null());
            let tmpl_testo = if tmpl.is_null() { String::new() } else { std::ffi::CStr::from_ptr(tmpl).to_string_lossy().into_owned() };
            if tmpl_testo.is_empty() && self.architettura() == "deepseek4" {
                // The local formatter supplies the chat markers for this architecture when
                // the weight file does not provide a template.
                (crate::service::deepseek4_conversation(msgs)?, false)
            } else if tmpl_testo.contains("<|turn>") {
                // The native formatter cannot render this template. The local formatter
                // trims message content and opens an empty reasoning channel.
                (crate::service::gemma4_conversation(msgs)?, false)
            } else {
                let mut buf = vec![0u8; lunghezza * 2 + 1024];
                let mut n = if tmpl.is_null() {
                    -1
                } else {
                    sys::llama_chat_apply_template(tmpl, c_msgs.as_ptr(), c_msgs.len(), true, buf.as_mut_ptr() as *mut c_char, buf.len() as i32)
                };
                if n > buf.len() as i32 {
                    buf.resize(n as usize + 16, 0);
                    n = sys::llama_chat_apply_template(tmpl, c_msgs.as_ptr(), c_msgs.len(), true, buf.as_mut_ptr() as *mut c_char, buf.len() as i32);
                }
                let ultimo_utente = msgs.last().map_or("", |m| m.text.as_str());
                if n >= 0 {
                    (String::from_utf8_lossy(&buf[..n as usize]).into_owned(), true)
                } else if tmpl_testo.contains("<start_of_turn>") {
                    eprintln!("[engine] chat template failed (rc={n}); using the Gemma 3 template");
                    let mut t = String::from("<bos>");
                    for m in msgs {
                        let r = if m.role == "assistant" { "model" } else { "user" };
                        t.push_str(&format!("<start_of_turn>{r}\n{}<end_of_turn>\n", m.text));
                    }
                    t.push_str("<start_of_turn>model\n");
                    (t, false)
                } else {
                    // With no usable history template, tokenize the last message as plain text.
                    eprintln!("[engine] chat template unavailable (rc={n}); using the raw prompt");
                    (ultimo_utente.to_string(), false)
                }
            }
        };
        // Suppress automatic BOS insertion when the rendered prompt already contains it.
        let aggiungi_speciali = unsafe { sys::llama_vocab_get_add_bos(self.vocab) } && !text.starts_with("<bos>");
        let t = self.tokenizza(&text, aggiungi_speciali)?;
        Ok((t, text, usato))
    }

    pub fn tokenizza(&self, text: &str, aggiungi_speciali: bool) -> Result<Vec<i32>, String> {
        let mut t = vec![0i32; text.len() + 16];
        unsafe {
            let mut n = sys::llama_tokenize(
                self.vocab,
                text.as_ptr() as *const c_char,
                text.len() as i32,
                t.as_mut_ptr(),
                t.len() as i32,
                aggiungi_speciali,
                true,
            );
            if n < 0 {
                t.resize((-n) as usize, 0);
                n = sys::llama_tokenize(
                    self.vocab,
                    text.as_ptr() as *const c_char,
                    text.len() as i32,
                    t.as_mut_ptr(),
                    t.len() as i32,
                    aggiungi_speciali,
                    true,
                );
            }
            if n < 0 {
                return Err("tokenization failed".into());
            }
            t.truncate(n as usize);
        }
        Ok(t)
    }

    /// Token bytes. Control tokens, including EOG, produce no text.
    pub fn pezzo(&self, token: i32) -> Vec<u8> {
        let mut buf = vec![0u8; 64];
        unsafe {
            let mut n = sys::llama_token_to_piece(self.vocab, token, buf.as_mut_ptr() as *mut c_char, buf.len() as i32, 0, false);
            if n < 0 {
                buf.resize((-n) as usize, 0);
                n = sys::llama_token_to_piece(self.vocab, token, buf.as_mut_ptr() as *mut c_char, buf.len() as i32, 0, false);
            }
            buf.truncate(n.max(0) as usize);
        }
        buf
    }
}

/// (n_layer, n_embd) from GGUF metadata, without loading weights.
pub fn info_gguf(path: &str) -> Result<(u32, u32), String> {
    let p = CString::new(path).map_err(|e| e.to_string())?;
    unsafe {
        let ip = sys::gguf_init_params { no_alloc: true, ctx: std::ptr::null_mut() };
        let g = sys::gguf_init_from_file(p.as_ptr(), ip);
        if g.is_null() {
            return Err(format!("{path}: GGUF illeggibile"));
        }
        let r = (|| {
            let k = sys::gguf_find_key(g, c"general.architecture".as_ptr());
            if k < 0 {
                return Err("missing general.architecture".to_string());
            }
            let arch = std::ffi::CStr::from_ptr(sys::gguf_get_val_str(g, k)).to_string_lossy().into_owned();
            let u32_di = |name: &str| -> Result<u32, String> {
                let c = CString::new(format!("{arch}.{name}")).unwrap();
                let k = sys::gguf_find_key(g, c.as_ptr());
                if k < 0 {
                    return Err(format!("missing {arch}.{name}"));
                }
                Ok(sys::gguf_get_val_u32(g, k))
            };
            Ok((u32_di("block_count")?, u32_di("embedding_length")?))
        })();
        sys::gguf_free(g);
        r
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            for &c in self.ctxs_mtp.iter().chain(&self.ctxs) {
                sys::llama_free(c);
            }
            if !self.model.is_null() {
                sys::llama_model_free(self.model);
            }
        }
    }
}
