use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyArray1};
use merkle_store::WeightStore as RustWeightStore;

#[pyclass]
struct WeightStore {
    inner: RustWeightStore,
}

#[pymethods]
impl WeightStore {
    #[new]
    fn new() -> Self {
        Self { inner: RustWeightStore::new() }
    }

    fn insert(&mut self, py: Python<'_>, tensor: PyReadonlyArray1<f32>) -> PyResult<PyObject> {
        let data = tensor.as_slice().expect("tensor insert failed at bridge");
        let hash = self.inner.insert(data);
        Ok(pyo3::types::PyBytes::new_bound(py, &hash).into())
    }

    fn get<'py>(&self, py: Python<'py>, hash: &[u8]) -> Option<Bound<'py, PyArray1<f32>>> {
        let hash_arr: [u8; 32] = hash.try_into().ok()?;
        let data = self.inner.get(&hash_arr)?;
        Some(PyArray1::from_slice_bound(py, data))
    }

    fn merkle_root(&self, py: Python<'_>) -> PyResult<PyObject> {
        let root = self.inner.merkle_root();
        Ok(pyo3::types::PyBytes::new_bound(py, &root).into())
    }

    fn merkle_root_hex(&self) -> String {
        let root = self.inner.merkle_root();
        root.iter().map(|b| format!("{:02x}", b)).collect()
    }

    fn set_layer_order(&mut self, hashes: Vec<Vec<u8>>) -> PyResult<()> {
        let mut converted = Vec::new();
        for h in hashes {
            let arr: [u8; 32] = h.try_into()
                .map_err(|_| pyo3::exceptions::PyValueError::new_err(
                        "each hash must be eaxactly 32 bytes"
                        ))?;

            converted.push(arr);
        }
        self.inner.set_layer_order(converted);
        Ok(())
    }
}

#[pymodule]
fn lattice_bridge(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<WeightStore>()?;
    Ok(())
}
