# Gradient-descent-for-STO

The code posted here is a code that optimizes the parameters of a spin-torque oscillator (STO) by gradient descent with automatic differentiation.

- experiment.py

  Main program

- Spin_neuralODE.py
  Program of the gradient descent with auto matic differentiation with torch.adjoint

  This program was made for the system identification.

  By changing the loss function and adding fc(1) and fc(2) to the set of the trained parameters, one can use this program for the MNIST task.

- coupledLlg.py

  Program of time Development of STO
