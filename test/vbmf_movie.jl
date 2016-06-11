using BayesianDataFusion
using MAT

## load and setup data
pkgdir = Pkg.dir("BayesianDataFusion")
print(pkgdir,"\n")
data   = matread("$pkgdir/data/movielens_1m.mat")
print("Loaded!")

## setup entities, assigning side information through optional argument F
users  = Entity("users")#,  F=data["Fu"]);
movies = Entity("movies")#, F=data["Fv"]);

## setup the relation between users and movies, data from sparse matrix data["X"]
## first element in '[users, movies]' corresponds to rows and second to columns of data["X"]
ratings = Relation(data["X"], "ratings", [users, movies], class_cut = 2.5);
setPrecision!(ratings,1.5)

## assign 500,000 of the observed ratings randomly to the test set
assignToTest!(ratings, 500_000)

## the model (with only one relation)
rd = RelationData(ratings)

# running the data
result = VBMF(rd;num_latent=10, niter=100, verbose = true)

