using BayesianDataFusion
using MAT

## load and setup data

X = read_matrix_market("../data/chembl-IC50-346targets.mm")
F = read_matrix_market("../data/chembl-IC50-compound-feat.mm")
print("Loaded!")

## setup entities, assigning side information through optional argument F
users  = Entity("comp",  F=F);
movies = Entity("prot")#, F=data["Fv"]);

## setup the relation between users and movies, data from sparse matrix data["X"]
## first element in '[users, movies]' corresponds to rows and second to columns of data["X"]
X.nzval -= mean(X.nzval)
ratings = Relation(X, "IC50", [users, movies], class_cut = 2.5);

## assign 500,000 of the observed ratings randomly to the test set
assignToTest!(ratings, 11_856)

## the model (with only one relation)
rd = RelationData(ratings)

# running the data
result = VBMF(rd;num_latent=30, niter=100, verbose = true)

