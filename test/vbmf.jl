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
Am = zeros(2,1)
F = spzeros(2,3)
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
Am = zeros(2,1)
Av = zeros(2,1)
F = spzeros(2,3)
Um = ones(1,3)
Uv = ones(1,3)
BayesianDataFusion.update_prior!(alpha, phi, Am, Av, Um, Uv, F::SparseMatrixCSC)
@test_approx_eq alpha 0.5

#Testing update_prior! with side-information
alpha = ones(1,1)
phi = ones(1,1)
Am = ones(2,1)*0.1
Av = ones(2,1)
F = spzeros(2,3)
F[1,1] = 1.0
F[1,2] = 1.0
F[1,3] = 1.0
Um = ones(1,3)
Uv = ones(1,3)
BayesianDataFusion.update_prior!(alpha, phi, Am, Av, Um, Uv, F::SparseMatrixCSC)
@test_approx_eq alpha 1/2.81
@test_approx_eq phi 1/1.01

#Testing update_latent! with side-information
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
Am = ones(2,1)*0.1
F = spzeros(2,3)
F[1,1] = 1.0
F[1,2] = 1.0
F[1,3] = 1.0
BayesianDataFusion.update_latent!(R, Um, Uv, alpha, tau, Vm, Vv, Am, F)
@test_approx_eq Uv      [1.0/7.0 1.0/7.0 1.0/7.0]
@test_approx_eq Um      [2.1/7.0 3.1/7.0 5.1/7.0]
@test_approx_eq R[1,3]  2.0 - 2.1/7.0
@test_approx_eq R[2,1]  3.0 - 3.1/7.0
@test_approx_eq R[3,1]  2.0 - 5.1/7.0
@test_approx_eq R[3,2]  3.0 - 5.1/7.0

#Testing update_link! 
Am = ones(2,1)*0.1
Av = ones(2,1)
F = spzeros(2,3)
F[1,1] = 1.0
F[2,1] = 1.0
F[1,2] = 1.0
F[2,2] = 1.0
F[1,3] = 1.0
F[2,3] = 1.0
Fnormsq = sum(F.^2,2)
alpha = ones(1,1)
phi = ones(1,1)
UR = ones(1,3) * 0.9
BayesianDataFusion.update_link!(UR, Am, Av, phi, alpha, F,Fnormsq)
@test_approx_eq Av [0.25 0.25]'
@test_approx_eq Am [0.75 0.2625]
@test_approx_eq UR [0.0875 0.0875 0.0875]

