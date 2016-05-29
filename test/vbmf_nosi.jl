using BayesianDataFusion
using Base.Test

Y  = spzeros(3,3)
Y[1,2] = -1.0
Y[1,3] =  1.0
Y[2,1] =  0.0
Y[2,2] =  0.0
Y[3,1] =  0.0
Y[3,3] = -1.0
rd = RelationData(Y, class_cut = 0.5)
#assignToTest!(rd.relations[1], 2)

# running the data
result = VBMF(rd;num_latent=1, verbose = true)

