# Performance of implicit restarting algorithms for Arnoldi for Large Sparse Matrices

This repository contains a Julia implementation of the **Implicitly Restarted Arnoldi Method (IRAM)** for computing a subset of eigenpairs of large sparse matrices.

## Project Structure

### `src/` (Core Implementation)

The core algorithm is separeted into different modules:

*   `Orthogonalization.jl`: Implements the Gram-Schimdt orthogonalization while storing the coefficients.
*   `Arnoldi.jl`: Implements the Arnoldi method.
*   `ImplicitRestart.jl`: Implements the implicit restart.
*   `BadRestart.jl`: Implements the naive restart.
*   `Eigenpairs.jl`: Implements the eigenpairs finding algorithms, namely the Arnoldi method, the naive restart and IRAM.

### `experiments/` (Validation and Testing)

*   `Arnoldi vs Naive vs IRAM.ipynb`: Compares the performance of the Arnoldi method, the naive restart and IRAM in terms of computation cost and memory allocation as a function of the Krylov subspace dimension for a large sparse matrix.
*   `IRAM Julia Libraries.ipynb`: Checks the residual obtained by IRAM and Krylov-Schur as implemented in the Julia libraries `Arpack` and `ArnoldiMethod`, respectively.
*   `IRAM vs Jacobi-Davidson.ipynb`: Compares the performance of IRAM and Jacobi-Davidson in terms of accuracy as a function of the Krylv subspace dimension and number of iterations for a large sparse matrix.
*   `Naive vs IRAM.ipynb`: Compares the performance of the naive restart and IRAM in terms of accuracy as a function of the Krylv subspace dimension and number of iteartions for a large sparse matrix.
*   `Validation.ipynb`: Validates the three implemented eigenpairs algorithms, namely the Arnoldi method, the naive restart and IRAM, by checking the accuracy as a function of the Krylov subspace dimension.

### `fig/` (Results)

*   `Validation/`: The plots from `Validation.ipynb`.
*   `Arnoldi vs Naive vs IRAM/`: The plots from `Arnoldi vs Naive vs IRAM.ipynb`.
*   `Naive vs IRAM/`: The plots from `Naive vs IRAM.ipynb`.
*   `IRAM vs Jacobi-Davidson/`: The plots from `IRAM vs Jacobi-Davidson.ipynb`.

## Software Requirements
This project is developed using **Julia 1.12.6**.
The following packages are required:
*   `ArnoldiMethod`
*   `Arpack`
*   `BenchmarkTools`
*   `Hungarian`
*   `JacobiDavidson`
*   `LaTeXStrings`
*   `LinearAlgebra`
*   `LinearMaps`
*   `MatrixDepot`
*   `Plots`
*   `ProgressMeter`