use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};
use bincode;
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GradientBlock{
    pub height: u64,
    pub prev_root: [u8; 32],
    pub new_root: [u8; 32],
    pub gradient_delta: HashMap<String, Vec<f32>>,
    pub timestamp_ms: u64,
    pub proposer_id: String,
}

impl GradientBlock{
    pub fn hash(&self) -> [u8; 32]{
        let bytes = bincode::serialize(self).expect("serialization failed");
        Sha256::new().chain_update(&bytes).finalize().into()
    }

    pub fn genesis() -> Self{
        Self{
            height: 0,
            prev_root: [0u8; 32],
            new_root: [0u8; 32],
            gradient_delta: HashMap::new(),
            timestamp_ms: 0,
            proposer_id: "genesis".to_string(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BlockError{
    #[error("height mismatch: expected {expected}, got {got}")]
    HeightMismatch { expected: u64, got: u64 },
    #[error("prev_root mismatch")]
    PrevRootMismatch,
    #[error("timestamp not after previous block")]
    TimestampNotMonotonic,
    #[error("new_root does not match computed root after applying delta")]
    RootMismatch,
}

pub struct ChainState{
    blocks: Vec<GradientBlock>,
}

impl ChainState{
    pub fn new(genesis: GradientBlock) -> Self{
        Self {blocks: vec![genesis]}
    }

    pub fn tip(&self) -> &GradientBlock {
        self.blocks.last().unwrap()
    }

    pub fn height(&self) -> u64 {
        self.tip().height
    }

    pub fn validate(&self, block: &GradientBlock) -> Result<(), BlockError>{
        let tip = self.tip();

        if block.height != tip.height + 1{
            return Err(BlockError::HeightMismatch {
                expected: tip.height + 1,
                got: block.height,
            });
        }

        if block.prev_root != tip.new_root{
            return Err(BlockError::PrevRootMismatch);
        }

        if block.timestamp_ms <= tip.timestamp_ms {
            return Err(BlockError::TimestampNotMonotonic);
        }

        // NOTE: full root verification (applying delta and checking new_root)
        // requires the WeightStore — that happens in the node layer, not here.
        // The chain validates structure; the node validates weight integrity.

        Ok(())
    }

    pub fn append(&mut self, block: GradientBlock) -> Result<(), BlockError>{
        self.validate(&block)?;
        self.blocks.push(block);
        Ok(())
    }

    pub fn root_at(&self, height: u64) -> Option<[u8; 32]>{
        self.blocks.get(height as usize).map(|b| b.new_root)
    }

    pub fn verify_full(&self) -> bool {
        for i in 1..self.blocks.len() {
            let prev = &self.blocks[i - 1];
            let curr = &self.blocks[i];
            if curr.prev_root != prev.new_root { return false;}
            if curr.height != prev.height + 1 { return false;}
        }
        true 
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn now_ms() -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64
    }

    fn make_block(height: u64, prev_root: [u8;32], new_root: [u8;32], ts: u64) -> GradientBlock {
        GradientBlock {
            height, prev_root, new_root,
            gradient_delta: HashMap::new(),
            timestamp_ms: ts,
            proposer_id: "test".to_string(),
        }
    }

    #[test]
    fn test_valid_chain() {
        let mut genesis = GradientBlock::genesis();
        genesis.new_root = [1u8; 32];
        genesis.timestamp_ms = 1000;
        let mut chain = ChainState::new(genesis);

        let block1 = make_block(1, [1u8;32], [2u8;32], 2000);
        let block2 = make_block(2, [2u8;32], [3u8;32], 3000);

        assert!(chain.append(block1).is_ok());
        assert!(chain.append(block2).is_ok());
        assert!(chain.verify_full());
        assert_eq!(chain.height(), 2);
    }

    #[test]
    fn test_rejects_wrong_prev_root() {
        let mut genesis = GradientBlock::genesis();
        genesis.new_root = [1u8; 32];
        genesis.timestamp_ms = 1000;
        let mut chain = ChainState::new(genesis);

        // wrong prev_root: says [9u8] but tip's new_root is [1u8]
        let bad_block = make_block(1, [9u8;32], [2u8;32], 2000);
        assert!(matches!(chain.append(bad_block), Err(BlockError::PrevRootMismatch)));
    }
}

