//! OpenAI-compatible HTTP server with agent sessions.
//!
//! One engine thread owns the GPU. Each loop it admits queued requests and runs one
//! batched step over every active request. A request with a `session_id` reuses that
//! session's sequence: the cached tokens that match the new prompt are kept, only the
//! rest is computed, and the sequence stays in memory after the request ends.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use serde_json::{json, Value};
use tokenizers::Tokenizer;

use crate::chat::{parse_tool_calls, split_reasoning, ChatTemplate};
use crate::engine::Engine;
use crate::kv::Seq;

/// What the engine loop sends back for a job: each new token, then the result.
enum Msg {
    Token(u32),
    Done(Result<Finished, String>),
}

struct Job {
    session: Option<String>,
    prompt: Vec<u32>,
    max_tokens: usize,
    ignore_eos: bool,
    reply: mpsc::Sender<Msg>,
}

struct Finished {
    output: Vec<u32>,
    prompt_tokens: usize,
    cached_tokens: usize,
    finish_reason: &'static str,
}

struct Active {
    job: Job,
    seq: Seq,
    output: Vec<u32>,
    cached_tokens: usize,
}

struct Session {
    seq: Seq,
    last_used: Instant,
    /// Until then the session's KV is not evicted for other sessions (agent in a tool call).
    pinned_until: Instant,
    /// Tool the last turn called, if it ended with a tool call.
    last_tool: Option<String>,
}

/// How long a session stays pinned after a turn.
#[derive(Clone, Copy, PartialEq)]
pub enum PinPolicy {
    /// Every turn: a fixed `pin_ttl`.
    Fixed,
    /// A turn the model ended without a tool call: no pin (the agent is probably done).
    /// Otherwise (tool call, or cut off by the length limit): the learned 90th-percentile
    /// wait for that key, x1.5 + 2 s, clamped to 1..300 s; `pin_ttl` until it has been seen.
    Adaptive,
}

/// Observed gaps between a turn that called a tool and the session's next request,
/// per tool name (most recent 64).
#[derive(Default)]
struct ToolWaits {
    gaps: HashMap<String, VecDeque<Duration>>,
}

impl ToolWaits {
    fn record(&mut self, tool: &str, gap: Duration) {
        let q = self.gaps.entry(tool.to_string()).or_default();
        if q.len() == 64 {
            q.pop_front();
        }
        q.push_back(gap);
    }

    fn p90(&self, tool: &str) -> Option<Duration> {
        let q = self.gaps.get(tool).filter(|q| !q.is_empty())?;
        let mut v: Vec<Duration> = q.iter().copied().collect();
        v.sort();
        Some(v[(v.len() * 9 / 10).min(v.len() - 1)])
    }

    fn pin_for(&self, tool: &str, default: Duration) -> Duration {
        match self.p90(tool) {
            Some(p) => (p.mul_f64(1.5) + Duration::from_secs(2)).clamp(Duration::from_secs(1), Duration::from_secs(300)),
            None => default,
        }
    }

    fn to_json(&self) -> Value {
        Value::Object(self.gaps.keys().map(|k| (k.clone(), json!({
            "samples": self.gaps[k].len(),
            "p90_s": self.p90(k).map(|d| d.as_secs_f64()),
        }))).collect())
    }
}

/// What the agent will probably wait on after this turn, as a key for learned wait times:
/// the first tool the output calls, "unfinished" if generation hit the length limit
/// (the agent likely continues), or None if the model ended the turn without a tool call
/// (the agent is probably done).
fn called_tool(tokenizer: &Tokenizer, tool_call_token: Option<u32>, output: &[u32], finish_reason: &str) -> Option<String> {
    if tool_call_token.is_some_and(|id| output.contains(&id)) {
        let text = tokenizer.decode(output, false).unwrap_or_default();
        let (_, calls) = parse_tool_calls(&text);
        return Some(calls.first().and_then(|c| c["function"]["name"].as_str()).unwrap_or("unknown").to_string());
    }
    (finish_reason != "stop").then(|| "unfinished".to_string())
}

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Most new tokens one step may process. Long prompts are split across steps (chunked
/// prefill) so a burst of new agent turns cannot exhaust GPU memory for activations.
/// Override with ENGINE_MAX_BATCH_TOKENS.
fn max_batch_tokens() -> usize {
    std::env::var("ENGINE_MAX_BATCH_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(4096)
}

/// Growth margin per resident session when deciding whether a new session fits: each
/// resident session is counted at its current length times this factor.
const GROWTH: f64 = 1.2;

/// Frees KV blocks until `need` are free, trimming idle sessions from the end, oldest
/// first. Only sessions whose pin has expired are touched unless `include_pinned`.
/// A session trimmed to nothing is dropped and stops being resident.
/// Returns true if `need` blocks are free afterwards.
fn free_up(engine: &mut Engine, sessions: &mut HashMap<String, Session>, resident: &mut HashSet<String>,
           need: usize, include_pinned: bool) -> bool {
    let bs = engine.pool.block_size;
    let now = Instant::now();
    let mut order: Vec<(Instant, String)> = sessions
        .iter()
        .filter(|(_, s)| include_pinned || s.pinned_until <= now)
        .map(|(k, s)| (s.last_used, k.clone()))
        .collect();
    order.sort();
    for (_, id) in order {
        let free = engine.pool.free_blocks();
        if free >= need {
            break;
        }
        let deficit = need - free;
        let s = sessions.get_mut(&id).unwrap();
        if s.seq.blocks.len() <= deficit {
            let mut s = sessions.remove(&id).unwrap();
            engine.pool.release(&mut s.seq);
            resident.remove(&id);
        } else {
            let keep_len = ((s.seq.blocks.len() - deficit) * bs).min(s.seq.tokens.len());
            engine.pool.truncate(&mut s.seq, keep_len);
        }
    }
    engine.pool.free_blocks() >= need
}

/// Blocks `a` still needs to compute its uncached tokens plus the next one.
fn blocks_needed(a: &Active, block_size: usize) -> usize {
    (a.seq.tokens.len() + 1).div_ceil(block_size).saturating_sub(a.seq.blocks.len())
}

/// The engine loop. Runs on its own thread until the job channel closes.
///
/// Memory policy (agent sessions first):
/// 1. A turn of a resident session (one whose KV is in the cache) is admitted before new
///    sessions. A new session is admitted only if every resident session at its current
///    length plus a growth margin, plus the new prompt, fits in the cache; otherwise it
///    waits. A session stops being resident when it is preempted or its pin expires.
/// 2. After a turn, the session stays pinned for `pin_ttl` so its KV survives the agent's
///    tool call. Only sessions past their pin are trimmed, oldest first, from the end.
/// 3. If running requests still cannot grow, pinned idle sessions are trimmed, then the most
///    recently admitted request is preempted (blocks freed, tokens kept, recomputed later).
/// Sessions idle for longer than `session_ttl` are freed.
#[allow(clippy::too_many_arguments)]
fn engine_loop(mut engine: Engine, jobs: mpsc::Receiver<Job>, eos: Vec<u32>, session_ttl: Duration, pin_ttl: Duration,
               stats: Arc<std::sync::Mutex<Value>>, tokenizer: Tokenizer, pin_policy: PinPolicy) {
    let tool_call_token = tokenizer.token_to_id("<tool_call>");
    let mut tool_waits = ToolWaits::default();
    let mut steps: u64 = 0;
    let mut preemptions: u64 = 0;
    let bs = engine.pool.block_size;
    let budget = max_batch_tokens();
    eprintln!("max batch tokens per step: {budget}");
    let mut sessions: HashMap<String, Session> = HashMap::new();
    let mut resident: HashSet<String> = HashSet::new();
    let mut active: Vec<Active> = Vec::new();
    let mut waiting: VecDeque<Active> = VecDeque::new();
    loop {
        // When idle, wait for work but wake up once a second to expire old sessions.
        let mut incoming: Vec<Job> = Vec::new();
        if active.is_empty() && waiting.is_empty() {
            match jobs.recv_timeout(Duration::from_secs(1)) {
                Ok(job) => incoming.push(job),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => return,
            }
        }
        incoming.extend(jobs.try_iter());

        let expired: Vec<String> = sessions.iter().filter(|(_, s)| s.last_used.elapsed() > session_ttl).map(|(k, _)| k.clone()).collect();
        for id in expired {
            let mut s = sessions.remove(&id).unwrap();
            engine.pool.release(&mut s.seq);
            resident.remove(&id);
        }

        for job in incoming {
            let mut seq = match job.session.as_ref().and_then(|id| sessions.remove(id)) {
                Some(s) => {
                    if let Some(tool) = &s.last_tool {
                        tool_waits.record(tool, s.last_used.elapsed());
                    }
                    s.seq
                }
                None => Seq::default(),
            };
            // Keep the part of the session that matches the new prompt. The last prompt
            // token is always recomputed so the step has something to produce logits from.
            let keep = common_prefix(&seq.tokens, &job.prompt).min(seq.cached).min(job.prompt.len() - 1);
            engine.pool.truncate(&mut seq, keep);
            seq.tokens.extend_from_slice(&job.prompt[keep..]);
            waiting.push_back(Active { cached_tokens: keep, seq, output: Vec::new(), job });
        }

        // A session stops being resident once its pin expires (its agent took too long).
        let now = Instant::now();
        for (id, sess) in sessions.iter() {
            if sess.pinned_until <= now {
                resident.remove(id);
            }
        }
        // Blocks the resident sessions are expected to need: current length plus growth margin.
        let footprint = |len: usize| ((len as f64 * GROWTH) as usize).div_ceil(bs);
        let mut resident_blocks: usize = sessions.iter().filter(|(id, _)| resident.contains(*id)).map(|(_, s)| footprint(s.seq.tokens.len())).sum::<usize>()
            + active.iter().chain(waiting.iter())
                .filter(|a| a.job.session.as_ref().is_some_and(|id| resident.contains(id)))
                .map(|a| footprint(a.seq.tokens.len()))
                .sum::<usize>();

        // Admission: resident sessions' turns first, then new sessions in arrival order,
        // a new one only if the resident sessions' expected size plus its own fits.
        let is_resident = |a: &Active, resident: &HashSet<String>| a.job.session.as_ref().is_some_and(|id| resident.contains(id));
        let mut order: Vec<usize> = (0..waiting.len()).filter(|&i| is_resident(&waiting[i], &resident)).collect();
        order.extend((0..waiting.len()).filter(|&i| !is_resident(&waiting[i], &resident)));
        let mut admitted: Vec<usize> = Vec::new();
        for i in order {
            let a = &waiting[i];
            let need = blocks_needed(a, bs);
            let returning = is_resident(a, &resident);
            let size = footprint(a.seq.tokens.len());
            let fits = if returning {
                free_up(&mut engine, &mut sessions, &mut resident, need, false)
                    || (active.is_empty() && free_up(&mut engine, &mut sessions, &mut resident, need, true))
            } else {
                let nothing_running = active.is_empty() && admitted.is_empty();
                (resident_blocks + size <= engine.pool.num_blocks || nothing_running)
                    && (free_up(&mut engine, &mut sessions, &mut resident, need, false)
                        || (nothing_running && free_up(&mut engine, &mut sessions, &mut resident, need, true)))
            };
            if fits {
                admitted.push(i);
                if let Some(id) = &waiting[i].job.session {
                    if resident.insert(id.clone()) {
                        resident_blocks += size;
                    }
                }
            } else if !returning {
                break; // keep new sessions in arrival order
            }
        }
        admitted.sort_unstable_by(|a, b| b.cmp(a));
        let mut batch: Vec<Active> = admitted.into_iter().map(|i| waiting.remove(i).unwrap()).collect();
        batch.reverse();
        active.extend(batch);

        if active.is_empty() {
            if let Some(mut a) = waiting.pop_front() {
                if blocks_needed(&a, bs) > engine.pool.num_blocks {
                    engine.pool.release(&mut a.seq);
                    let msg = format!("request needs more KV blocks than the cache has ({})", engine.pool.num_blocks);
                    eprintln!("rejecting request: {msg}");
                    let _ = a.job.reply.send(Msg::Done(Err(msg)));
                } else {
                    waiting.push_front(a);
                }
            }
            continue;
        }

        // Make sure every running request can grow by one token this step.
        loop {
            let need: usize = active.iter().map(|a| blocks_needed(a, bs)).sum();
            if free_up(&mut engine, &mut sessions, &mut resident, need, false)
                || free_up(&mut engine, &mut sessions, &mut resident, need, true)
            {
                break;
            }
            if active.len() == 1 {
                break;
            }
            let mut a = active.pop().unwrap();
            engine.pool.release(&mut a.seq);
            a.cached_tokens = 0;
            if let Some(id) = &a.job.session {
                resident.remove(id);
            }
            waiting.push_front(a);
            preemptions += 1;
        }

        steps += 1;
        *stats.lock().unwrap() = json!({
            "steps": steps, "preemptions": preemptions, "active": active.len(), "waiting": waiting.len(),
            "resident": resident.len(), "idle_sessions": sessions.len(),
            "free_blocks": engine.pool.free_blocks(), "total_blocks": engine.pool.num_blocks,
            "tool_waits": tool_waits.to_json(),
        });

        let mut seqs: Vec<&mut Seq> = active.iter_mut().map(|a| &mut a.seq).collect();
        let next = match engine.step(&mut seqs, budget) {
            Ok(next) => next,
            Err(e) => {
                eprintln!("engine step failed: {e}");
                for mut a in active.drain(..) {
                    engine.pool.release(&mut a.seq);
                    let _ = a.job.reply.send(Msg::Done(Err(e.to_string())));
                }
                continue;
            }
        };

        let mut still = Vec::new();
        for (mut a, tok) in active.drain(..).zip(next) {
            // No token yet: this request's prompt is still being processed in chunks.
            let Some(tok) = tok else {
                still.push(a);
                continue;
            };
            let is_eos = !a.job.ignore_eos && eos.contains(&tok);
            if !is_eos {
                a.output.push(tok);
                let _ = a.job.reply.send(Msg::Token(tok));
            }
            let finish = if is_eos { Some("stop") } else if a.output.len() >= a.job.max_tokens { Some("length") } else { None };
            let Some(finish_reason) = finish else {
                still.push(a);
                continue;
            };
            if is_eos {
                // The EOS token was appended but not computed; drop it so the session ends clean.
                a.seq.tokens.pop();
            }
            let done = Finished {
                output: std::mem::take(&mut a.output),
                prompt_tokens: a.job.prompt.len(),
                cached_tokens: a.cached_tokens,
                finish_reason,
            };
            match a.job.session.take() {
                Some(id) => {
                    let now = Instant::now();
                    let last_tool = called_tool(&tokenizer, tool_call_token, &done.output, done.finish_reason);
                    let pin = match (pin_policy, &last_tool) {
                        (PinPolicy::Fixed, _) => pin_ttl,
                        (PinPolicy::Adaptive, None) => Duration::ZERO,
                        (PinPolicy::Adaptive, Some(tool)) => tool_waits.pin_for(tool, pin_ttl),
                    };
                    sessions.insert(id, Session { seq: a.seq, last_used: now, pinned_until: now + pin, last_tool });
                }
                None => engine.pool.release(&mut a.seq),
            }
            let _ = a.job.reply.send(Msg::Done(Ok(done)));
        }
        active = still;
    }
}

struct AppState {
    jobs: std::sync::Mutex<mpsc::Sender<Job>>,
    stats: Arc<std::sync::Mutex<Value>>,
    tokenizer: Tokenizer,
    template: ChatTemplate,
    model_name: String,
}

type HttpError = (u16, String);

/// Renders and tokenizes the request, queues it, and returns the engine's message stream.
fn submit(st: &AppState, body: &Value) -> Result<mpsc::Receiver<Msg>, HttpError> {
    let bad = |e: String| (400, e);
    let tools = body.get("tools").filter(|t| !t.is_null());
    let prompt = st.template.render(&body["messages"], tools).map_err(|e| bad(e.to_string()))?;
    let ids = st.tokenizer.encode(prompt, false).map_err(|e| bad(e.to_string()))?.get_ids().to_vec();
    let (reply, rx) = mpsc::channel();
    let job = Job {
        session: body["session_id"].as_str().map(String::from),
        prompt: ids,
        max_tokens: body["max_tokens"].as_u64().or(body["max_completion_tokens"].as_u64()).unwrap_or(1024) as usize,
        ignore_eos: body["ignore_eos"].as_bool().unwrap_or(false),
        reply,
    };
    st.jobs.lock().unwrap().send(job).map_err(|e| (500, e.to_string()))?;
    Ok(rx)
}

fn usage(done: &Finished) -> Value {
    json!({
        "prompt_tokens": done.prompt_tokens,
        "completion_tokens": done.output.len(),
        "total_tokens": done.prompt_tokens + done.output.len(),
        "prompt_tokens_details": {"cached_tokens": done.cached_tokens},
    })
}

fn complete(st: &AppState, rx: mpsc::Receiver<Msg>) -> Result<Value, HttpError> {
    let done = loop {
        match rx.recv().map_err(|e| (500, e.to_string()))? {
            Msg::Token(_) => {}
            Msg::Done(r) => break r.map_err(|e| (500, e))?,
        }
    };
    let text = st.tokenizer.decode(&done.output, true).map_err(|e| (500, e.to_string()))?;
    let (reasoning, text) = split_reasoning(&text);
    let (content, tool_calls) = parse_tool_calls(&text);
    let mut message = json!({"role": "assistant", "content": content});
    if let Some(r) = reasoning {
        message["reasoning_content"] = json!(r);
    }
    let finish_reason = if tool_calls.is_empty() { done.finish_reason } else { "tool_calls" };
    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls);
    }
    Ok(json!({
        "id": "chatcmpl-engine",
        "object": "chat.completion",
        "model": st.model_name,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage(&done),
    }))
}

/// Streams OpenAI-style SSE chunks: text deltas as they decode, then a final chunk with
/// finish_reason and usage. Deltas are raw text (no reasoning or tool-call splitting yet).
fn stream(st: Arc<AppState>, rx: mpsc::Receiver<Msg>, out: mpsc::Sender<Vec<u8>>) {
    let sse = |v: Value| format!("data: {v}\n\n").into_bytes();
    let chunk = |delta: Value, finish: Value| json!({
        "id": "chatcmpl-engine", "object": "chat.completion.chunk", "model": st.model_name,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    });
    let mut tokens = Vec::new();
    let mut sent = String::new();
    // The role rides on the first real text delta, so time-to-first-chunk is time-to-first-token.
    let mut first = true;
    loop {
        // While the request waits in the queue, send an SSE comment every 15 s so proxies
        // do not close an idle connection. Clients ignore comment lines.
        let msg = match rx.recv_timeout(Duration::from_secs(15)) {
            Ok(m) => m,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if out.send(b": keep-alive\n\n".to_vec()).is_err() {
                    break;
                }
                continue;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };
        match msg {
            Msg::Token(t) => {
                tokens.push(t);
                // Decode everything so far and send only the new, complete text.
                let text = st.tokenizer.decode(&tokens, true).unwrap_or_default();
                if text.len() > sent.len() && !text.ends_with('\u{FFFD}') && text.starts_with(sent.as_str()) {
                    let delta = text[sent.len()..].to_string();
                    sent = text;
                    let delta = if first { json!({"role": "assistant", "content": delta}) } else { json!({"content": delta}) };
                    first = false;
                    let _ = out.send(sse(chunk(delta, Value::Null)));
                }
            }
            Msg::Done(Ok(done)) => {
                let mut last = chunk(json!({}), json!(done.finish_reason));
                last["usage"] = usage(&done);
                let _ = out.send(sse(last));
                break;
            }
            Msg::Done(Err(e)) => {
                let _ = out.send(sse(json!({"error": {"message": e}})));
                break;
            }
        }
    }
    let _ = out.send(b"data: [DONE]\n\n".to_vec());
}

/// `Read` over a channel of byte chunks, so tiny_http can send a chunked response.
struct ChannelReader {
    rx: mpsc::Receiver<Vec<u8>>,
    buf: Vec<u8>,
    pos: usize,
}

impl std::io::Read for ChannelReader {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        while self.pos >= self.buf.len() {
            match self.rx.recv() {
                Ok(b) => {
                    self.buf = b;
                    self.pos = 0;
                }
                Err(_) => return Ok(0),
            }
        }
        let n = out.len().min(self.buf.len() - self.pos);
        out[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

fn header(name: &str, value: &str) -> tiny_http::Header {
    tiny_http::Header::from_bytes(name.as_bytes(), value.as_bytes()).unwrap()
}

fn respond(req: tiny_http::Request, status: u16, body: String) {
    let _ = req.respond(tiny_http::Response::from_string(body).with_status_code(status).with_header(header("Content-Type", "application/json")));
}

fn handle(st: Arc<AppState>, mut req: tiny_http::Request) {
    let path = req.url().split('?').next().unwrap_or("").to_string();
    match (req.method().as_str(), path.as_str()) {
        ("GET", "/health") => respond(req, 200, "\"ok\"".into()),
        ("GET", "/profile") => respond(req, 200, crate::prof::take().to_string()),
        ("GET", "/stats") => {
            let body = st.stats.lock().unwrap().to_string();
            respond(req, 200, body)
        }
        ("GET", "/v1/models") => {
            let body = json!({"object": "list", "data": [{"id": st.model_name, "object": "model"}]});
            respond(req, 200, body.to_string())
        }
        ("POST", "/v1/chat/completions") => {
            let mut raw = String::new();
            if let Err(e) = req.as_reader().read_to_string(&mut raw) {
                return respond(req, 400, json!({"error": {"message": e.to_string()}}).to_string());
            }
            let body = match serde_json::from_str::<Value>(&raw) {
                Ok(b) => b,
                Err(e) => return respond(req, 400, json!({"error": {"message": e.to_string()}}).to_string()),
            };
            let rx = match submit(&st, &body) {
                Ok(rx) => rx,
                Err((code, msg)) => return respond(req, code, json!({"error": {"message": msg}}).to_string()),
            };
            if body["stream"].as_bool().unwrap_or(false) {
                let (tx, bytes) = mpsc::channel();
                let st2 = Arc::clone(&st);
                std::thread::spawn(move || stream(st2, rx, tx));
                let reader = ChannelReader { rx: bytes, buf: Vec::new(), pos: 0 };
                let response = tiny_http::Response::new(200.into(), vec![header("Content-Type", "text/event-stream")], reader, None, None);
                let _ = req.respond(response);
            } else {
                match complete(&st, rx) {
                    Ok(v) => respond(req, 200, v.to_string()),
                    Err((code, msg)) => respond(req, code, json!({"error": {"message": msg}}).to_string()),
                }
            }
        }
        _ => respond(req, 404, json!({"error": {"message": "not found"}}).to_string()),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn serve(engine: Engine, tokenizer: Tokenizer, template: ChatTemplate, eos: Vec<u32>, model_name: String, port: u16,
             session_ttl: Duration, pin_ttl: Duration, pin_policy: PinPolicy) -> Result<()> {
    let (tx, rx) = mpsc::channel();
    let stats = Arc::new(std::sync::Mutex::new(json!({})));
    let loop_stats = Arc::clone(&stats);
    let loop_tokenizer = tokenizer.clone();
    std::thread::spawn(move || engine_loop(engine, rx, eos, session_ttl, pin_ttl, loop_stats, loop_tokenizer, pin_policy));
    let state = Arc::new(AppState { jobs: std::sync::Mutex::new(tx), stats, tokenizer, template, model_name });
    let server = tiny_http::Server::http(("0.0.0.0", port)).map_err(anyhow::Error::msg)?;
    println!("listening on 0.0.0.0:{port}");
    // One thread per request: each blocks until the engine loop finishes its job.
    for req in server.incoming_requests() {
        let st = Arc::clone(&state);
        std::thread::spawn(move || handle(st, req));
    }
    Ok(())
}
