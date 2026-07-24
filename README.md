# 🌊 Lattice Boltzmann Fluid Simulation (D2Q9)

![Von Karman Vortex Street](von_karman_simulation.gif)

## Project Description
This project simulates 2D fluid flow around a rigid obstacle using the **Lattice Boltzmann Method (LBM)**. It demonstrates the formation of the **Von Kármán Vortex Street**, a classic phenomenon in fluid dynamics where repeating swirling vortices are caused by the unsteady separation of flow of a fluid around blunt bodies.

This simulation was built from scratch in Python to understand the underlying physics of transport equations and collision operators.

## How it Works
Instead of solving the macroscopic Navier-Stokes equations, LBM models the fluid as fictitious particles performing consecutive propagation and collision processes over a discrete lattice mesh.

* **Model:** D2Q9 (2 Dimensions, 9 Velocities)
* **Collision Operator:** BGK (Bhatnagar-Gross-Krook)
* **Reynolds Number:** ~150 (Unsteady Laminar Flow)

## Stack
* **Python 3.14**
* **NumPy:** For high-performance vectorized matrix operations.
* **Matplotlib:** For real-time visualization and rendering.

## Usage
1.  Run the static analysis:
    ```bash
    python lbm_solver.py
    ```
2.  Generate the animation:
    ```bash
    python lbm_video.py
    ```

---
Developed by Enio
