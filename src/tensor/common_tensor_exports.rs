use pyo3::prelude::*;

/// Distribution for random value of a tensor.
#[pyclass]
#[derive(Clone, Debug)]
pub enum Distribution {
    // Default,
    Bernoulli(f64),
    Uniform(f64, f64),
    Normal(f64, f64),
}
