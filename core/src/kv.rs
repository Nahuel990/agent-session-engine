//! Paged KV cache pool and sequences.
//!
//! A `Seq` owns its blocks for as long as it lives. That is the core of the engine:
//! an agent session keeps its `Seq` between turns, so its history stays in the
//! cache and only new tokens are ever computed.

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};

use crate::model::Config;

pub struct KvPool {
    /// One (key, value) pair per layer, each `(num_blocks, block_size, num_kv_heads, head_dim)`.
    pub caches: Vec<(Tensor, Tensor)>,
    pub block_size: usize,
    pub num_blocks: usize,
    free: Vec<u32>,
}

impl KvPool {
    pub fn new(cfg: &Config, num_blocks: usize, block_size: usize, dtype: DType, dev: &Device) -> Result<Self> {
        let shape = (num_blocks, block_size, cfg.num_key_value_heads, cfg.head_dim);
        let caches = (0..cfg.num_hidden_layers)
            .map(|_| Ok((Tensor::zeros(shape, dtype, dev)?, Tensor::zeros(shape, dtype, dev)?)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { caches, block_size, num_blocks, free: (0..num_blocks as u32).rev().collect() })
    }

    pub fn free_blocks(&self) -> usize {
        self.free.len()
    }

    /// Grows `seq`'s block table so it can hold `len` tokens.
    pub fn reserve(&mut self, seq: &mut Seq, len: usize) -> Result<()> {
        let needed = len.div_ceil(self.block_size);
        while seq.blocks.len() < needed {
            match self.free.pop() {
                Some(b) => seq.blocks.push(b),
                None => bail!("KV cache full: need {needed} blocks, have {} free", self.free.len()),
            }
        }
        Ok(())
    }

    /// Keeps only the first `len` tokens of `seq` and frees the blocks past them.
    /// Stale K/V left in the last kept block is never read: attention stops at the
    /// context length, and new tokens overwrite those slots.
    pub fn truncate(&mut self, seq: &mut Seq, len: usize) {
        seq.tokens.truncate(len);
        seq.cached = seq.cached.min(len);
        let keep = len.div_ceil(self.block_size);
        if seq.blocks.len() > keep {
            self.free.extend(seq.blocks.drain(keep..));
        }
    }

    pub fn release(&mut self, seq: &mut Seq) {
        self.free.extend(seq.blocks.drain(..));
        seq.cached = 0;
    }
}

#[derive(Default)]
pub struct Seq {
    /// Every token of the sequence so far.
    pub tokens: Vec<u32>,
    /// Physical KV blocks, in order.
    pub blocks: Vec<u32>,
    /// How many leading tokens already have their K/V in the cache.
    pub cached: usize,
}

impl Seq {
    pub fn new(tokens: Vec<u32>) -> Self {
        Self { tokens, ..Default::default() }
    }

    pub fn uncached(&self) -> usize {
        self.tokens.len() - self.cached
    }
}
