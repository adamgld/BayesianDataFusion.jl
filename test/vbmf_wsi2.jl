# Implementation of exactly the same test setup implemented in Python version

using BayesianDataFusion
using Base.Test
srand(1234)
F = sprandn(2,100,1.0)
G = spzeros(2,2)
A = ones(2,1)
A[1,1] =  1.0
A[2,1] = -2.0

B = zeros(2,1) 

U = A' * F 
V = [1, -1]'

Y  = sparse(U' * V + 0.05 * randn(100,2))

entity1 = Entity("E1", F=F')
entity2 = Entity("E2")
rel = Relation(Y, "rel", [entity1, entity2])
setPrecision!(rel, 20)
rd = RelationData(rel)
assignToTest!(rd.relations[1], 100)

# running the data
result = VBMF(rd;num_latent=2, niter=100, verbose = true)
#result = macau(rd; burnin = 100, psamples = 100, num_latent = 2,)
