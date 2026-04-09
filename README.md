# Physics-Informed Neural Network (PINN) Solver for the 2D Poisson Equation

This project implements a **Physics-Informed Neural Network (PINN)** in Python using **PyTorch** to solve the **2D Poisson equation** on a unit square domain.

The solution is validated against a classical **Finite Element Method (FEM)** solver, and controlled experiments are performed to compare network architectures and optimization strategies.

The project demonstrates how **machine learning can be integrated into scientific computing workflows** for solving physics-based engineering problems.

---

# Motivation

Traditional numerical methods such as the **Finite Element Method (FEM)** are widely used in **computational mechanics**, **heat transfer**, and **engineering simulation**.

However, modern engineering workflows increasingly integrate **machine learning**, **scientific AI**, and **data-driven modeling**.

**Physics-Informed Neural Networks (PINNs)** provide an alternative approach in which the governing physics is built directly into the training objective.

This project was developed to:

- understand how neural networks can solve PDEs
- compare PINNs with a trusted FEM reference
- explore **GPU-based scientific computing**
- study the effect of model architecture and optimization strategy on solution accuracy
- demonstrate reproducible numerical experiments for physics-based machine learning

---

# What is a Physics-Informed Neural Network (PINN)?

A **Physics-Informed Neural Network** is a neural network trained not only on data, but also on the **physical laws** that govern the system.

Instead of learning from labeled input-output pairs, the model is trained to satisfy the **differential equation residual** and the **boundary conditions**.

In simple terms:

```text
Neural Network + Governing Equations + Boundary Conditions = Physics-Informed Model
```

PINNs are useful in:

- **heat transfer**
- **fluid dynamics**
- **structural mechanics**
- **computational physics**
- **inverse problems**
- **scientific machine learning**

---