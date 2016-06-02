using BayesianDataFusion
using Base.Test

F = sprand(2,20,1.0)
G = sprand(2,20,1.0)
A = ones(2,2)
A[1,1] = 2.0
A[1,2] = 1.5
A[2,1] = 1.5
A[2,2] = 2.0

B = ones(2,2) 
B[1,1] = 1.0
B[1,2] = 0.5
B[2,1] = 0.5
B[2,2] = 1.0

U = A' * F + randn(2,20)/5.0
V = B' * G + randn(2,20)/5.0

Y  = sparse(U' * V + randn(20,20)/2.0)

print("Original:\n")
print("U=",U,"\n")
print("V=",V,"\n")

entity1 = Entity("E1")#, F=F)
entity2 = Entity("E2")#, F=G)
rel = Relation(Y, "rel", [entity1, entity2])
rd = RelationData(rel)
assignToTest!(rd.relations[1], 100)

# running the data
result = VBMF(rd;num_latent=2, niter=100, verbose = true)

