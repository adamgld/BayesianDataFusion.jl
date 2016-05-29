using BayesianDataFusion
using Base.Test

#Testing update_latent! without side-information
I = [1,2,3,3]
J = [3,1,1,2]
V = [1.0,2.0,1.0,2.0]
R = sparse(I,J,V)
Um = ones(1,3)
Uv = ones(1,3)
Vm = ones(1,3)
Vv = ones(1,3)
tau = 1.0
alpha = ones(1,1)
Am = zeros(1,3)
F = spzeros(1,3)
BayesianDataFusion.update_latent!(R, Um, Uv, alpha, tau, Vm, Vv, Am, F)
@test_approx_eq Uv      [1.0/7.0 1.0/7.0 1.0/7.0]
@test_approx_eq Um      [2.0/7.0 3.0/7.0 5.0/7.0]
@test_approx_eq R[1,3]  2.0 - 2.0/7.0
@test_approx_eq R[2,1]  3.0 - 3.0/7.0
@test_approx_eq R[3,1]  2.0 - 5.0/7.0 
@test_approx_eq R[3,2]  3.0 - 5.0/7.0

#Testing update_prior! without side-information
alpha = ones(1,1)
phi = ones(1,1)
Am = zeros(1,3)
Av = zeros(1,3)
F = spzeros(1,3)
Um = ones(1,3)
Uv = ones(1,3)
BayesianDataFusion.update_prior!(alpha, phi, Am, Av, Um, Uv, F::SparseMatrixCSC)
@test_approx_eq alpha 0.5
