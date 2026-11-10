mod chat;
mod engine;
mod kv;
mod model;
mod prof;
mod server;

use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use engine::Engine;
use kv::{KvPool, Seq};
use model::{Config, Qwen3};

const BLOCK_SIZE: usize = 64;

fn load(model_id: &str, num_blocks: usize) -> Result<(Engine, Tokenizer)> {
    let repo = hf_hub::api::sync::Api::new()?.model(model_id.to_string());
    let cfg: Config = serde_json::from_slice(&std::fs::read(repo.get("config.json")?)?)?;
    let tokenizer = Tokenizer::from_file(repo.get("tokenizer.json")?).map_err(anyhow::Error::msg)?;
    let files = match repo.get("model.safetensors.index.json") {
        Ok(index) => {
            let index: serde_json::Value = serde_json::from_slice(&std::fs::read(index)?)?;
            let mut names: Vec<String> = index["weight_map"]
                .as_object()
                .unwrap()
                .values()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();
            names.sort();
            names.dedup();
            names.iter().map(|f| repo.get(f)).collect::<Result<Vec<_>, _>>()?
        }
        Err(_) => vec![repo.get("model.safetensors")?],
    };
    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0)?;
    #[cfg(not(feature = "cuda"))]
    let device = Device::Cpu;
    let dtype = DType::BF16;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? };
    let model = Qwen3::load(cfg.clone(), vb, dtype, &device)?;
    let pool = KvPool::new(&cfg, num_blocks, BLOCK_SIZE, dtype, &device)?;
    Ok((Engine { model, pool, dtype, device, last_logits: None }, tokenizer))
}

fn encode(tok: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tok.encode(text, false).map_err(anyhow::Error::msg)?.get_ids().to_vec())
}

fn generate(engine: &mut Engine, seq: &mut Seq, n: usize) -> Result<(Vec<u32>, usize)> {
    let mut out = Vec::new();
    let mut computed = 0;
    for _ in 0..n {
        computed += seq.uncached();
        out.push(engine.step(&mut [seq], usize::MAX)?[0].unwrap());
    }
    Ok((out, computed))
}

/// Teacher-forced check: feed `reference` one token at a time after `prompt` and report,
/// per step, our top-2 tokens with log-probs and our log-prob of the reference token.
fn teacher_forced(engine: &mut Engine, prompt: &[u32], reference: &[u32]) -> Result<serde_json::Value> {
    let mut seq = Seq::new(prompt.to_vec());
    let mut steps = Vec::new();
    for &want in reference {
        engine.step(&mut [&mut seq], usize::MAX)?;
        seq.tokens.pop();
        seq.tokens.push(want);
        let lp = candle_nn::ops::log_softmax(&engine.last_logits.as_ref().unwrap().get(0)?, 0)?.to_vec1::<f32>()?;
        let mut idx: Vec<usize> = (0..lp.len()).collect();
        idx.sort_by(|&a, &b| lp[b].total_cmp(&lp[a]).then(a.cmp(&b)));
        steps.push(serde_json::json!({"top1": idx[0], "top1_lp": lp[idx[0]], "top2": idx[1], "top2_lp": lp[idx[1]], "ref_lp": lp[want as usize]}));
    }
    engine.pool.release(&mut seq);
    Ok(serde_json::Value::Array(steps))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("serve") {
        // engine serve <model_id> <port> <num_blocks> <session_ttl_secs> <pin_ttl_secs>
        let model_id = args.get(2).cloned().unwrap_or_else(|| "Qwen/Qwen3-0.6B".into());
        let port: u16 = args.get(3).map(|p| p.parse()).transpose()?.unwrap_or(8000);
        let num_blocks: usize = args.get(4).map(|n| n.parse()).transpose()?.unwrap_or(512);
        let ttl: u64 = args.get(5).map(|n| n.parse()).transpose()?.unwrap_or(600);
        let pin: u64 = args.get(6).map(|n| n.parse()).transpose()?.unwrap_or(30);
        let (engine, tok) = load(&model_id, num_blocks)?;
        let repo = hf_hub::api::sync::Api::new()?.model(model_id.clone());
        let template = chat::ChatTemplate::from_tokenizer_config(&std::fs::read(repo.get("tokenizer_config.json")?)?)?;
        let gen: serde_json::Value = serde_json::from_slice(&std::fs::read(repo.get("generation_config.json")?)?)?;
        let eos: Vec<u32> = match &gen["eos_token_id"] {
            serde_json::Value::Array(v) => v.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect(),
            v => v.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
        };
        let (policy, name) = match std::env::var("ENGINE_PIN_POLICY").as_deref() {
            Ok("adaptive") => (server::PinPolicy::Adaptive, "adaptive"),
            Ok("capped") => (server::PinPolicy::Capped, "capped"),
            _ => (server::PinPolicy::Fixed, "fixed"),
        };
        eprintln!("pin policy: {name}");
        return server::serve(engine, tok, template, eos, "llm".into(), port,
            std::time::Duration::from_secs(ttl), std::time::Duration::from_secs(pin), policy);
    }

    let model_id = args.get(1).cloned().unwrap_or_else(|| "Qwen/Qwen3-0.6B".into());
    let (mut engine, tok) = load(&model_id, 512)?;

    if let Some(path) = args.get(2) {
        // Reference file from vllm_ref.py: {"prompt_ids": [...], "tokens": [...]}
        let r: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        let ids = |k: &str| -> Vec<u32> { r[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect() };
        let steps = teacher_forced(&mut engine, &ids("prompt_ids"), &ids("tokens"))?;
        println!("{}", serde_json::json!({"check": "teacher_forced", "steps": steps}));
        return Ok(());
    }

    // 1. Greedy generation, for a token-by-token comparison with vLLM.
    let prompt = "The capital of France is";
    let prompt_ids = encode(&tok, prompt)?;
    let mut seq = Seq::new(prompt_ids.clone());
    let mut out = Vec::new();
    let mut top5 = Vec::new();
    for _ in 0..8 {
        out.push(engine.step(&mut [&mut seq], usize::MAX)?[0].unwrap());
        let logprobs = candle_nn::ops::log_softmax(&engine.last_logits.as_ref().unwrap().get(0)?, 0)?;
        let mut v: Vec<(usize, f32)> = logprobs.to_vec1::<f32>()?.into_iter().enumerate().collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        top5.push(v[..5].iter().map(|(t, p)| (*t, (*p * 1000.0).round() / 1000.0)).collect::<Vec<_>>());
    }
    let (rest, _) = generate(&mut engine, &mut seq, 24)?;
    out.extend(rest);
    println!("{}", serde_json::json!({"check": "greedy", "prompt": prompt, "prompt_ids": prompt_ids,
        "tokens": out, "top5_first8": top5, "text": tok.decode(&out, false).map_err(anyhow::Error::msg)?}));
    engine.pool.release(&mut seq);

    // 2. Session reuse: turn 2 on a live cache must match a fresh run over the full history.
    let turn1 = encode(&tok, "Agent: I will list the files.\nTool: README.md main.rs\nAgent:")?;
    let turn2 = encode(&tok, "\nTool: done\nAgent:")?;
    let mut live = Seq::new(turn1.clone());
    let (_, computed1) = generate(&mut engine, &mut live, 16)?;
    let history = live.tokens.clone();
    live.tokens.extend(&turn2);
    let (live_out, computed2) = generate(&mut engine, &mut live, 16)?;

    let mut fresh = Seq::new([history.as_slice(), turn2.as_slice()].concat());
    let (fresh_out, computed_fresh) = generate(&mut engine, &mut fresh, 16)?;
    println!("{}", serde_json::json!({
        "check": "session",
        "match": live_out == fresh_out,
        "turn1_tokens_computed": computed1,
        "turn2_tokens_computed_live": computed2,
        "turn2_tokens_computed_fresh": computed_fresh,
        "live": live_out, "fresh": fresh_out,
    }));
    Ok(())
}
