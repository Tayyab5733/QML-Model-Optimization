# Quantum Machine Learning with PennyLane, JAX, and Optax

This project demonstrates a simple quantum machine learning model built using:
- **PennyLane** for quantum circuits,
- **JAX** for auto-differentiation and JIT compilation,
- **Optax** for optimization with the Adam optimizer.

## 🧠 Model Description

The model consists of:
1. **Data Encoding**: Classical data is encoded into quantum states via rotation gates (RY).
2. **Trainable Ansatz**: A layered quantum circuit with RX, RY gates and CNOTs connects qubits and introduces trainable parameters.
3. **Measurement**: The circuit outputs the sum of Pauli-Z expectation values on each wire.
4. **Prediction**: Output is adjusted by a trainable bias to match target values.
5. **Loss Function**: Mean squared error between predictions and targets is minimized.

## ⚙️ Optimization

Two training approaches are implemented:
- **Standard Loop**: Regular Python loop with gradient computation.
- **JIT-Optimized Loop**: Uses `jax.jit` and `jax.lax.fori_loop` for speed.

Both approaches are benchmarked using Python's `timeit` module to compare performance.

## 📊 Dependencies

Install the required packages via:

```bash
pip install pennylane jax optax
