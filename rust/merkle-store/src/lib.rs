use sha2::{Sha256, Digest};
use std::collections::HashMap;

pub fn hash_tensor(data: &[f32]) -> [u8; 32] {
    
    let bytes = bytemuck::cast_slice(data);
    Sha256::new().chain_update(bytes).finalize().into()

}

pub fn compute_merkel_root(leaves: &[[u8;32]]) -> [u8;32] {
    if leaves.is_empty() {
        return [0u8; 32];
    }
    if leaves.len() == 1 {
        return leaves[0];
    }

    let mut current: Vec<[u8;32]> = leaves.to_vec();
    while current.len() > 1 {
        let mut next = Vec::new();
        let mut i = 0;
        while i < current.len() {
            let left = current[i];

            let right = if i + 1 < current.len() {
                current[i + 1]
            }else{
                current[i]
            };
            
            let mut hasher = Sha256::new();
            hasher.update(left);
            hasher.update(right);
            next.push(hasher.finalize().into());
            i += 2;
        }
        current = next;
    }
    current[0]
}

#[derive(Default)]
pub struct WeightStore{
    data: HashMap<[u8; 32], Vec<u8>>,
    layer_hashes: Vec<[u8; 32]>,
}

impl WeightStore{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, tensor: &[f32]) -> [u8; 32]{
        let hash = hash_tensor(tensor);
        let bytes = bytemuck::cast_slice(tensor).to_vec();
        self.data.insert(hash, bytes);
        hash
    }

    pub fn get(&self, hash: &[u8; 32]) -> Option<&[f32]>{
        self.data.get(hash).map(|b| bytemuck::cast_slice(b))
    }

    pub fn set_layer_order(&mut self, hashes: Vec<[u8; 32]>){
        self.layer_hashes = hashes
    }

    pub fn merkle_root(&self) -> [u8; 32]{
        compute_merkel_root(&self.layer_hashes)
    }

    pub fn apply_delta(
        &mut self, 
        layer_hash: &[u8; 32],
        delta: &[f32]
        ) -> Result<[u8; 32], String>{

        let current = self.get(layer_hash).ok_or("hash not found in store")?.to_vec();

        if current.len() != delta.len(){
            return Err(format!(
            "delta length mismatch: {} vs {}",
                current.len(), delta.len()
                    ))
        }

        let updated: Vec<f32> = current
            .iter()
            .zip(delta.iter())
            .map(|(w, d)| w + d)
            .collect();

        Ok(self.insert(&updated))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_retrieve() {
        let mut store = WeightStore::new();
        let tensor = vec![1.0f32, 2.0, 3.0, 4.0];
        let hash = store.insert(&tensor);
        let retrieved = store.get(&hash).unwrap();
        assert_eq!(retrieved, tensor.as_slice());
    }

    #[test]
    fn test_merkle_root_changes_on_weight_update() {
        let mut store = WeightStore::new();
        let t1 = vec![1.0f32; 64];
        let t2 = vec![2.0f32; 64];
        let h1 = store.insert(&t1);
        let h2 = store.insert(&t2);
        store.set_layer_order(vec![h1, h2]);

        let root_before = store.merkle_root();

        // mutate t1 slightly
        let mut t1_modified = t1.clone();
        t1_modified[0] = 99.0;
        let h1_new = store.insert(&t1_modified);
        store.set_layer_order(vec![h1_new, h2]);

        let root_after = store.merkle_root();
        assert_ne!(root_before, root_after, "root must change when weights change");
    }

    #[test]
    fn test_apply_delta() {
        let mut store = WeightStore::new();
        let weights = vec![1.0f32; 4];
        let hash = store.insert(&weights);
        let delta = vec![0.1f32; 4];
        let new_hash = store.apply_delta(&hash, &delta).unwrap();
        let updated = store.get(&new_hash).unwrap();
        assert!((updated[0] - 1.1).abs() < 1e-6);
    }
}




