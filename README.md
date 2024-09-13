# QUBO Benchmarking: Classical vs Quantum Solvers

Welcome to the **QUBO Benchmarking** project repository! This repository contains the code, datasets, and benchmarking results comparing the performance of classical and quantum solvers on a variety of Quadratic Unconstrained Binary Optimization (QUBO) problems.

## Project Overview

In this project, we aim to evaluate and compare the **speed**, **accuracy**, and **scalability** of various solvers for QUBO problems, focusing on both classical approaches and quantum solvers (e.g., D-Wave quantum annealing, QAOA).

### What is a QUBO?

A **Quadratic Unconstrained Binary Optimization (QUBO)** problem is a type of mathematical optimization problem where the objective is to minimize a quadratic function of binary variables. QUBO problems are NP-hard and appear in various domains such as finance, logistics, machine learning, and more.

### Solvers Tested

- **Classical Solvers**: Gurobi, QuBowl, and approximation algorithms
- **Quantum Solvers**: D-Wave's quantum annealer, IBM Qiskit (QAOA)

## Repository Structure

- `instances/`  
  Contains a collection of QUBO problem instances used in our benchmarking. These instances are categorized into:
  - Portfolio Optimization
  - Max-Cut Problems
  - Scheduling Problems
  - Randomly Generated QUBOs

- `benchmark/`  
  Contains the code and scripts used for running the benchmarks on the solvers. Includes configurations for both classical and quantum solvers.

- `results/`  
  Contains benchmarking results, including performance metrics (speed, accuracy, etc.) for each solver across the QUBO instances.

## Getting Started

### Prerequisites

To run the benchmarking code, you will need:

- **Python 3.x**
- **Quantum Computing Frameworks** (e.g., D-Wave Ocean SDK, IBM Qiskit)
- **Classical Optimization Libraries** (e.g., Gurobi, SciPy)

### Installing Dependencies

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/qubo-benchmarking.git
   cd qubo-benchmarking
