export VBMF

function update_latent!(R::SparseMatrixCSC, Um, Uv, alpha, tau::Float64,Vm, Vv, Am::Matrix{Float64}, F::SparseMatrixCSC)
  K,I = size(Um)
  _,J = size(Vm)
  Js,_ = findn(R)
  AmF = Am' * F
  for i = 1:I
    for k = 1:K
      tmp = Um[k,i]
      Uvtmp = 0
      for idx in R.colptr[i]:(R.colptr[i+1]-1)
          j = Js[idx]
          Uvtmp += Vm[k,j].^2 + Vv[k,j]
      end
      Uv[k,i] = 1/(alpha[k] + tau * Uvtmp)

      theta = sum(alpha[k] * AmF[k,i]) 

      for idx in R.colptr[i]:(R.colptr[i+1]-1)
          j = Js[idx]
          theta += tau * (R.nzval[idx] + Um[k,i]*Vm[k,j]).*Vm[k,j]
      end
#      print("theta = ", theta,"\n")

      Um[k,i] = Uv[k,i] * theta

      for idx in R.colptr[i]:(R.colptr[i+1]-1)
        j = Js[idx]
        R.nzval[idx] -= (Um[k,i]-tmp)*Vm[k,j]
      end
    end
  end 
end

function update_link!(UR, Am::Matrix{Float64}, Av::Matrix{Float64}, phi, alpha, Ft::SparseMatrixCSC,Fnormsq)
  M,K = size(Am)
  for k = 1:K
    for m = 1:M
      tmp = Am[m,k]
      Av[m,k] = 1/(phi[k] + alpha[k] * Fnormsq[m])
      temp = 0.0
      for idx = Ft.colptr[m]:(Ft.colptr[m+1]-1)
	Fim = Ft.nzval[idx]
        i = Ft.rowval[idx]
        temp += UR[k,i] * Fim + Am[m,k] * Fim*Fim
       # @time Am[m,k] = Av[m,k] * alpha[k] * sum((UR[k,:] + Am[m,k] * Ftmt)* Ftm)
      end
      Am[m,k] = Av[m,k] * alpha[k] * temp
      for idx = Ft.colptr[m]:(Ft.colptr[m+1]-1)
        Fim = Ft.nzval[idx]
        i = Ft.rowval[idx]
	UR[k,i] -= (Am[m,k] - tmp) *Fim
       # @time UR[k,:] = UR[k,:] - (Am[m,k] - tmp) * Ftmt
      end
#      print("UR=", UR,"\n")
    end
  end
end

function update_link_naive!(Um, Am::Matrix{Float64}, Av::Matrix{Float64}, phi, alpha, F::SparseMatrixCSC,Fnormsq)
  M,K = size(Am)
  _,I = size(F)
  for m = 1:M
    Av[m,:] = 1./ (phi + alpha * Fnormsq[m])
    for k = 1:K
      tmp = 0
      for i = 1:I
        diff = Um[k,i]
	for e = 1:M
	  if e!=m
            diff -= F[e,i] * Am[e,k]
	  end
	end
        tmp += diff * F[k,i]
      end
      Am[m,k] = Av[m,k] *tmp * alpha[k]
    end
  end
end

function update_prior!(alpha, phi, Am, Av, Um, Uv, F::SparseMatrixCSC)
  K,I = size(Um)
  M,_ = size(Am)
  for k=1:K
    alpha[k] = I./sum((Um[k,:] - Am[:,k]' * F).^2 + Uv[k,:] +  Av[:,k]' * (F.^2))
    phi[k] = (M + 0.001) ./ (0.001 + sum(Am[:,k].^2 + Av[:,k]))
  end
end

function pred(r::Relation, probe_vec, U, V)
  N = size(probe_vec,1)
  ret = zeros(N)
  for i = 1:N
    ret[i] = dot(U[:,r.test_vec[i,1]],V[:,r.test_vec[i,2]]) + r.model.mean_value
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
  I = data.entities[1].count
  J = data.entities[2].count
  X = sparse(Is, Js,data.relations[1].data.df[3], I,J)
   
  rel.model.mean_value = mean(rel.data.df[3])
  print("Internal mean:" , rel.model.mean_value,"\n")

  #Initialization
  K=num_latent
  tau = rel.model.alpha

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
  F = data.entities[1].F'
  Ft = F'
  G = data.entities[2].F'
  Gt = G'
  if isempty(F)
    F = spzeros(1,I)
  end
  if isempty(G)
    G = spzeros(1,J)
  end

  print("Size(F)=",size(F),"\n")
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
      V[idx] = X[i,j] - dot(Um[:,i],Vm[:,j]) - rel.model.mean_value
      idx=idx+1
  end
  R = sparse(Js,Is,V,J,I)
  UR = Um - Am' * F
  VR = Vm - Bm' * G
  haveTest = numTest(rel) > 0

  for t=1:niter #Iteration
    update_latent!(R,Um,Uv,alpha,tau,Vm,Vv,Am,F)
    R = R'
    update_latent!(R,Vm,Vv, beta,tau,Um,Uv,Bm,G)
    R = R'
  UR = Um - Am' * F
  VR = Vm - Bm' * G

    if !isempty(data.entities[1].F)
	update_link!(UR, Am, Av, phiA, alpha, Ft, Fnormsq)
	#update_link_naive!(Um, Am, Av, phiA, alpha, F, Fnormsq)
    end
    if !isempty(data.entities[2].F)
        update_link!(VR, Bm, Bv, phiB, beta, Gt, Gnormsq)
	#update_link_naive!(Vm, Bm, Bv, phiB, beta, G, Gnormsq)
    end
 #   print("----->", alpha,"\n")
 #   print("----->", beta,"\n")

    update_prior!(alpha, phiA, Am, Av, Um, Uv, F)
    update_prior!(beta, phiB, Bm, Bv, Vm, Vv, G)

    probe_rat = pred(rel, rel.test_vec, Um,Vm)
    #print ("Calc=", Um[:,rel.test_vec[1,1]]' * Vm[:,rel.test_vec[1,2]],"\n")
    #print("Prob_rat = ", probe_rat[1],"\n")
    #clamped_rat = isempty(clamp) ?probe_rat :makeClamped(probe_rat, clamp)
    rmse = haveTest ? sqrt(mean( (rel.test_vec[:,end] - probe_rat) .^ 2 )) : NaN
    roc = haveTest ? AUC_ROC(rel.test_label, -vec(probe_rat)) : NaN    
    if verbose
      print("RMSE=",rmse,"\tROC=",roc,"\t|Res|=",sqrt(mean((R.*R).nzval)),"\t|E[U]|=",vecnorm(Um),"\t|E[V]|=",vecnorm(Vm),"\t|E[A]|=",vecnorm(Am),"\t|E[B]|=",vecnorm(Bm))
      print("\t|phiA|=",vecnorm(phiA))
      print("\t|phiB|=",vecnorm(phiB))
      print("\n")
    end
  end

 # print("Predicted:\n")
 # print(Um,"\n")
 # print(Vm,"\n")

 #print("Am = ", Am, "\n")

 #print(phiA)
end
