export VBMF

function update_latent!(R::SparseMatrixCSC, Um, Uv, alpha, tau::Float64,Vm, Vv, Am::Matrix{Float64}, F::SparseMatrixCSC)
  K,I = size(Um)
  _,J = size(Vm)
  Js,_ = findn(R)
  for i = 1:I
    for k = 1:K
      tmp = Um[k,i]
      Uv[k,i] = 1/(alpha[k] + tau * sum(Vm[k,:].^2 + Vv[k,:]))
      theta = sum(alpha[k] * Am[:,k]' * F[:,i]) 

      for idx in R.colptr[i]:(R.colptr[i+1]-1)
          j = Js[idx]
          theta += tau * (R.nzval[idx] + Um[k,i]*Vm[k,j]).*Vm[k,j]
      end

      Um[k,i] = Uv[k,i] * theta

      for idx in R.colptr[i]:(R.colptr[i+1]-1)
        j = Js[idx]
        R.nzval[idx] -= (Um[k,i]-tmp)*Vm[k,j]
      end
    end
  end 
end

function update_link!(UR, Am::Matrix{Float64}, Av::Matrix{Float64}, phi, alpha, F::SparseMatrixCSC,Fnormsq)
  M,K = size(Am)
  for k = 1:K
    for m = 1:M
      tmp = Am[m,k]
      Av[m,k] = 1/(phi[k] + alpha[k] * Fnormsq[m])
      Am[m,k] = Av[m,k] * alpha[k] * sum((UR[k,:] + Am[m,k] * F[m,:])* F[m,:]')
      UR[k,:] = UR[k,:] - (Am[m,k] - tmp) * F[m,:]
    end
  end
end

function update_prior!(alpha, phi, Am, Av, Um, Uv, F::SparseMatrixCSC)
  K,I = size(Um)
  M,_ = size(Am)
  for k=1:K
    alpha[k] = I./sum((Um[k,:] - Am[:,k]' * F).^2 + Uv[k,:] +  Av[:,k]' * (F.^2))
    phi[k] = M./sum(Am[:,k].^2 + Av[:,k])
  end
end

function pred(r::Relation, probe_vec, U, V)
  N = size(probe_vec,1)
  ret = zeros(N)
  for i = 1:N
    ret[i] = dot(U[:,r.test_vec[i,1]],V[:,r.test_vec[i,2]])
  end
  return ret
end

function VBMF(data::RelationData;
	      num_latent::Int = 10,
	      verbose::Bool   = true,
	      niter::Int      = 100)

  #Processing RelationData structure
  rel = data.relations[1]
  Is = rel.data.df[1]
  Js = rel.data.df[2]
  X = sparse(Is, Js,data.relations[1].data.df[3])

  #Initialization
  K=num_latent
  I=maximum(Is)
  J=maximum(Js)
  tau = 1.50 #
  tau = 50.0

  n = Normal(0,1)
  Um = 0.3 * rand(n,K,I)
  Uv = 4 * ones(K,I)
  Vm = 0.3 * rand(n,K,J)
  Vv = 4 * ones(K,J)
  alpha = ones(K)
  beta = ones(K)
  phiA = ones(K)
  phiB = ones(K)

  #SI related
  F = data.entities[1].F
  G = data.entities[2].F
  if ! data.entities[1].use_FF
    F = spzeros(1,I)
  end
  if ! data.entities[2].use_FF
    G = spzeros(1,J)
  end
  Mf = size(F,1)
  Mg = size(G,1)
  Am = 0.3 * rand(n,Mf,K)
  Av = 4 * ones(Mf,K)
  Bm = 0.3 * rand(n,Mg,K)
  Bv = 4 * ones(Mg,K)
  
  Fnormsq = sum(F.^2,2)
  Gnormsq = sum(G.^2,2)

  V=zeros(nnz(X))
  for idx in 1:length(Is)
      i = Is[idx]
      j = Js[idx]
      V[idx] = X[i,j] - dot(Um[:,i],Vm[:,j])
      idx=idx+1
  end
  R = sparse(Js,Is,V)
  UR = Um - Am' * F
  VR = Vm - Bm' * G
  haveTest = numTest(rel) > 0

  for t=1:niter #Iteration
    update_latent!(R,Um,Uv,alpha,tau,Vm,Vv,Am,F)
    R = R'
    update_latent!(R,Vm,Vv, beta,tau,Um,Uv,Bm,G)
    R = R'
    update_link!(UR, Am, Av, alpha, phiA, F, Fnormsq)
    update_link!(VR, Bm, Bv, beta, phiB, G, Gnormsq)
    update_prior!(alpha, phiA, Am, Av, Um, Uv, F)
    update_prior!(beta, phiB, Bm, Bv, Vm, Vv, G)

    probe_rat = pred(rel, rel.test_vec, Um,Vm)
    #print ("Calc=", Um[:,rel.test_vec[1,1]]' * Vm[:,rel.test_vec[1,2]],"\n")
    #print("Prob_rat = ", probe_rat[1],"\n")
    rmse = haveTest ? sqrt(mean( (rel.test_vec[:,end] - probe_rat) .^ 2 )) : NaN
    
    if verbose
      print("RMSE=",rmse,"\t|Res|=",sqrt(mean((R.*R).nzval)))#,"\t|E[U]|=",vecnorm(Um),"\t|E[V]|=",vecnorm(Vm),"|",alpha,"|",beta)
      print("\tstd=",std(rel.test_vec[:,end]))
      print("\n")
    end
  end

 # print("Predicted:\n")
 # print(Um,"\n")
 # print(Vm,"\n")
end
