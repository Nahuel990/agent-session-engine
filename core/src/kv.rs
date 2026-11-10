//! Paged KV cache pool with prefix sharing across sequences.
//!
//! Every full block a sequence has computed gets a hash chained from the previous block's
//! hash, so a hash identifies the whole token prefix up to the end of that block. An index
//! maps hashes to physical blocks. A new sequence starts by taking the longest run of
//! indexed blocks matching its tokens, from any sequence or session, so identical prompt
//! prefixes (system prompt, tool definitions) are computed and stored once.
//!
//! Blocks are reference counted. A block nobody holds goes to an LRU list but stays in the
//! index until the space is needed, so it can still be reused. Indexed or shared blocks are
//! never written: a sequence only writes into its own blocks past its cached prefix.

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};

use crate::model::Config;

pub struct KvPool {
    /// One (key, value) pair per layer, each `(num_blocks, block_size, num_kv_heads, head_dim)`.
    pub caches: Vec<(Tensor, Tensor)>,
    pub block_size: usize,
    pub num_blocks: usize,
    /// Never-indexed blocks nobody holds.
    free: Vec<u32>,
    refcount: Vec<u32>,
    /// Prefix hash of an indexed block.
    hash_of: Vec<Option<u64>>,
    index: HashMap<u64, u32>,
    /// Indexed blocks nobody holds, oldest release first: tick -> block.
    lru: BTreeMap<u64, u32>,
    lru_tick: Vec<Option<u64>>,
    tick: u64,
}

fn chain_hash(prev: u64, tokens: &[u32]) -> u64 {
    let mut h = DefaultHasher::new();
    prev.hash(&mut h);
    tokens.hash(&mut h);
    h.finish()
}

impl KvPool {
    pub fn new(cfg: &Config, num_blocks: usize, block_size: usize, dtype: DType, dev: &Device) -> Result<Self> {
        let shape = (num_blocks, block_size, cfg.num_key_value_heads, cfg.head_dim);
        let caches = (0..cfg.num_hidden_layers)
            .map(|_| Ok((Tensor::zeros(shape, dtype, dev)?, Tensor::zeros(shape, dtype, dev)?)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            caches,
            block_size,
            num_blocks,
            free: (0..num_blocks as u32).rev().collect(),
            refcount: vec![0; num_blocks],
            hash_of: vec![None; num_blocks],
            index: HashMap::new(),
            lru: BTreeMap::new(),
            lru_tick: vec![None; num_blocks],
            tick: 0,
        })
    }

    /// Blocks that can be handed out now (free plus evictable cached blocks).
    pub fn free_blocks(&self) -> usize {
        self.free.len() + self.lru.len()
    }

    /// Cached blocks nobody holds (still reusable by prefix match).
    pub fn cached_idle_blocks(&self) -> usize {
        self.lru.len()
    }

    fn alloc(&mut self) -> Option<u32> {
        let b = match self.free.pop() {
            Some(b) => b,
            None => {
                let (_, b) = self.lru.pop_first()?;
                self.lru_tick[b as usize] = None;
                if let Some(h) = self.hash_of[b as usize].take() {
                    self.index.remove(&h);
                }
                b
            }
        };
        self.refcount[b as usize] = 1;
        Some(b)
    }

    fn acquire(&mut self, b: u32) {
        if let Some(t) = self.lru_tick[b as usize].take() {
            self.lru.remove(&t);
        }
        self.refcount[b as usize] += 1;
    }

    fn release_block(&mut self, b: u32) {
        let rc = &mut self.refcount[b as usize];
        *rc -= 1;
        if *rc > 0 {
            return;
        }
        if self.hash_of[b as usize].is_some() {
            self.tick += 1;
            self.lru.insert(self.tick, b);
            self.lru_tick[b as usize] = Some(self.tick);
        } else {
            self.free.push(b);
        }
    }

    /// Grows `seq`'s block table so it can hold `len` tokens.
    pub fn reserve(&mut self, seq: &mut Seq, len: usize) -> Result<()> {
        let needed = len.div_ceil(self.block_size);
        while seq.blocks.len() < needed {
            match self.alloc() {
                Some(b) => seq.blocks.push(b),
                None => bail!("KV cache full: need {needed} blocks, have {} available", self.free_blocks()),
            }
        }
        Ok(())
    }

    /// Indexes every full block of `seq` whose tokens are all computed, so other sequences
    /// can share it. A block whose prefix is already indexed elsewhere stays private.
    pub fn register(&mut self, seq: &mut Seq) {
        let bs = self.block_size;
        while seq.block_hashes.len() < seq.cached / bs {
            let i = seq.block_hashes.len();
            let prev = if i == 0 { 0 } else { seq.block_hashes[i - 1] };
            let h = chain_hash(prev, &seq.tokens[i * bs..(i + 1) * bs]);
            seq.block_hashes.push(h);
            let b = seq.blocks[i];
            if self.hash_of[b as usize].is_none() && !self.index.contains_key(&h) {
                self.index.insert(h, b);
                self.hash_of[b as usize] = Some(h);
            }
        }
    }

    /// Takes the longest run of indexed blocks matching `tokens` (full blocks only) and
    /// returns a sequence holding them, with those tokens marked cached.
    pub fn match_prefix(&mut self, tokens: &[u32]) -> Seq {
        let bs = self.block_size;
        let mut seq = Seq::default();
        let mut prev = 0;
        for chunk in tokens.chunks_exact(bs) {
            let h = chain_hash(prev, chunk);
            let Some(&b) = self.index.get(&h) else { break };
            self.acquire(b);
            seq.blocks.push(b);
            seq.block_hashes.push(h);
            seq.tokens.extend_from_slice(chunk);
            prev = h;
        }
        seq.cached = seq.tokens.len();
        seq
    }

    /// Keeps at most the first `len` tokens of `seq` and returns how many it kept. If `len`
    /// ends inside a block that is shared or indexed, it is rounded down to that block's
    /// start: new tokens must not be written into a block others may read.
    pub fn truncate(&mut self, seq: &mut Seq, len: usize) -> usize {
        let bs = self.block_size;
        let mut len = len.min(seq.tokens.len());
        if len % bs != 0 {
            if let Some(&b) = seq.blocks.get(len / bs) {
                if self.refcount[b as usize] > 1 || self.hash_of[b as usize].is_some() {
                    len = len / bs * bs;
                }
            }
        }
        seq.tokens.truncate(len);
        seq.cached = seq.cached.min(len);
        seq.block_hashes.truncate(len / bs);
        let keep = len.div_ceil(bs);
        let dropped: Vec<u32> = seq.blocks.drain(keep.min(seq.blocks.len())..).collect();
        for b in dropped.into_iter().rev() {
            self.release_block(b);
        }
        len
    }

    /// Drops all of `seq`'s blocks. Indexed ones stay reusable until evicted; the tail is
    /// released first, so it is evicted first.
    pub fn release(&mut self, seq: &mut Seq) {
        let blocks: Vec<u32> = seq.blocks.drain(..).collect();
        for b in blocks.into_iter().rev() {
            self.release_block(b);
        }
        seq.block_hashes.clear();
        seq.cached = 0;
    }
}

#[derive(Default)]
pub struct Seq {
    /// Every token of the sequence so far.
    pub tokens: Vec<u32>,
    /// Physical KV blocks, in order.
    pub blocks: Vec<u32>,
    /// Chained prefix hash of each full computed block, in order.
    pub block_hashes: Vec<u64>,
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
