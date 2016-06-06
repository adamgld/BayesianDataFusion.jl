using BayesianDataFusion
using Base.Test

#STRONGER
#Testing update_latent! without side-information
I = [1,2,3,3]
J = [3,1,1,2]
V = [1.0,2.0,1.0,2.0]
R = sparse(J,I,V)
Um = [0.1, 0.2, 0.3]'
Uv = ones(1,3)
Vm = [0.4, 0.5, 0.6]'
Vv = [0.5, 0.2, 0.5]'
tau = 2.0
alpha = 1.1 * ones(1,1)
Am = zeros(2,1)
F = spzeros(2,3)
BayesianDataFusion.update_latent!(R, Um, Uv, alpha, tau, Vm, Vv, Am, F)
@test_approx_eq Uv      [1.0/2.82 1.0/2.42 1.0/3.32]
@test_approx_eq Um      [1.272/2.82 1.664/2.42 3.046/3.32]
@test_approx_eq R[3,1]  1.0 - (1.272/2.82 - 0.1) * 0.6
@test_approx_eq R[1,2]  2.0 - (1.664/2.42 - 0.2) * 0.4
@test_approx_eq R[1,3]  1.0 - (3.046/3.32 - 0.3) * 0.4
@test_approx_eq R[2,3]  2.0 - (3.046/3.32 - 0.3) * 0.5

#Testing update_link!
Am = ones(2,1)
Am[1,1] = 0.3
Am[2,1] = 0.5
Av = ones(2,1)
F = spzeros(2,3)
F[1,1] = 1.0
F[2,1] = 2.0
F[1,2] = 3.0
F[2,2] = 4.0
F[1,3] = 5.0
F[2,3] = 6.0
Fnormsq = sum(F.^2,2)
alpha = 1.1 * ones(1,1)
phi = 1.5 * ones(1,1)
UR = [0.9,0.8,0.7]'
BayesianDataFusion.update_link!(UR, Am, Av, phi, alpha, F,Fnormsq)
@test_approx_eq Av [0.025 1/63.1]'
@test_approx_eq Am [0.025*19.03 32.4137/63.1]
@test_approx_eq UR [0.72425 0.27275 -0.17875] - (32.4137/63.1 - 0.5) * [2.0 4.0 6.0]

#Testing update_prior! with side-information
alpha = 1.1 * ones(1,1)
phi = 1.5 * ones(1,1)
Am = ones(2,1)
Am[1,1] = 0.3
Am[2,1] = 0.5
Av = ones(2,1)
Av[1,1] = 0.1
Av[2,1] = 0.05
F = spzeros(2,3)
F[1,1] = 1.0
F[2,1] = 2.0
F[1,2] = 3.0
F[2,2] = 4.0
F[1,3] = 5.0
F[2,3] = 6.0
Um = [0.9 0.8 0.7]
Uv = [0.4 0.5 0.6]
BayesianDataFusion.update_prior!(alpha, phi, Am, Av, Um, Uv, F::SparseMatrixCSC)
@test_approx_eq alpha 3/26.81
@test_approx_eq phi 2/0.49

