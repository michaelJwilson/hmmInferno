import numpy as np
from numpy import linalg as LA

n_states = 7

diag_rate = 1. - 10.**-5.
offdiag_rate = (1. - diag_rate) / (n_states - 1.)

# print(diag_rate, offdiag_rate)

transfer = np.ones(shape=(n_states,n_states))
transfer -= np.eye(n_states)
transfer *= offdiag_rate

transfer += np.diag(n_states * [diag_rate])

# print(transfer)

# print(eigenvalues)
# print(eigenvectors)

start_probs = np.ones(shape=(n_states, 1))

current_probs = np.dot(transfer, start_probs)

exp = np.ones(shape=(n_states, 1), dtype=float)

# NB P(zm) * t + (K-1) * P(!zm) * (1-t) / (K-1).
assert np.allclose(current_probs, exp)
              
eigenvalues, eigenvectors = LA.eigh(transfer)

idx = np.argmax(eigenvalues)

# print(eigenvectors[6])

nsteps = 10_000
nstep_transfer = LA.matrix_power(transfer, nsteps)

# print(transfer)
result = np.dot(transfer, transfer)

print(transfer)
print()
print(nstep_transfer)
