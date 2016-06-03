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


