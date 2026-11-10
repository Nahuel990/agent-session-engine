//! Optional per-section timing, on when ENGINE_PROFILE=1.
//!
//! Each timed section waits for the GPU before and after, so the numbers are real GPU time
//! but the engine runs slower while profiling. Use only for diagnosis.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::Result;
use candle_core::Device;

static TOTALS: Mutex<BTreeMap<&'static str, (Duration, u64)>> = Mutex::new(BTreeMap::new());

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ENGINE_PROFILE").is_ok_and(|v| v == "1"))
}

/// Runs `f`, adding its GPU-synchronized duration to section `name` when profiling is on.
pub fn time<T>(dev: &Device, name: &'static str, f: impl FnOnce() -> Result<T>) -> Result<T> {
    if !enabled() {
        return f();
    }
    dev.synchronize()?;
    let start = Instant::now();
    let out = f()?;
    dev.synchronize()?;
    let mut totals = TOTALS.lock().unwrap();
    let entry = totals.entry(name).or_default();
    entry.0 += start.elapsed();
    entry.1 += 1;
    Ok(out)
}

/// Adds a count without timing (e.g. tokens processed).
pub fn count(name: &'static str, n: u64) {
    if enabled() {
        TOTALS.lock().unwrap().entry(name).or_default().1 += n;
    }
}

/// Returns `{section: {"ms": total, "calls": n}}` and resets the totals.
pub fn take() -> serde_json::Value {
    let mut totals = TOTALS.lock().unwrap();
    let out = totals
        .iter()
        .map(|(k, (d, n))| (k.to_string(), serde_json::json!({"ms": d.as_secs_f64() * 1000.0, "calls": n})))
        .collect::<serde_json::Map<_, _>>();
    totals.clear();
    serde_json::Value::Object(out)
}
