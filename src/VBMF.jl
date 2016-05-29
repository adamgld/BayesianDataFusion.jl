export VBMF

function update_latent!(R::SparseMatrixCSC, Um, Uv, alpha, tau::Float64,Vm, Vv, Am::Matrix{Float64}, F::SparseMatrixCSC)
  K,I = size(Um)
  _,J = size(Vm)
  Is,Js = findn(R)
  for i = 1:I
    for k = 1:K
      tmp = Um[k,i]
      Uv[k,i] = 1/(alpha[k] + tau * sum(Vm[k,:].^2 + Vv[k,:]))
      theta = sum(alpha[k] * Am[:,k]' * F[:,i]) 
      if i in Is
        for j in Js[Is .==i]
          theta += tau * (R[i,j] + Um[k,i]*Vm[k,j]).*Vm[k,j]
        end
      end
      Um[k,i] = Uv[k,i] * theta
      if i in Is
        for j in Js[Is .== i]
          R[i,j] = R[i,j] - (Um[k,i]-tmp)*Vm[k,j]
        end
      end
    end
  end 
end

function update_link!(UR, Am::Matrix{Float64}, Av::Matrix{Float64}, phi, alpha, F::SparseMatrixCSC)
end

function update_prior!(alpha, phi, Am, Av, Um, Uv, F::SparseMatrixCSC)
  K,I = size(Um)
  for k=1:K
    alpha[k] = I./sum((Uv[k,:] - Am[:,k]' * F).^2 + Uv[k,:] +  Av[:,k]' * (F.^2))
    #phi = 
  end
end

function VBMF(data::RelationData;
	      num_latent::Int = 10,
	      verbose::Bool   = true,
	      niter::Int      = 100)

  #Processing RelationData structure
  Is = data.relations[1].data.df[1]
  Js = data.relations[1].data.df[2]
  X=sparse(Is, Js,data.relations[1].data.df[3])
  #Initialization
  K=num_latent
  I=maximum(Is)
  J=maximum(Js)
  tau = 5.0
  n = Normal(0,1)
  Um = rand(n,K,I)
  Uv = ones(K,I)
  Vm = rand(n,K,J)
  Vv = ones(K,J)
  alpha = ones(K)
  beta = ones(K)

  #Dummies
  phiA = zeros(K)
  phiB = zeros(K)
  Am = zeros(3,K)
  Av = zeros(3,K)
  F = spzeros(3,I)
  Bm = zeros(3,K)
  Bv = zeros(3,K)
  G = spzeros(3,J)
  #TODO: Initialize Am; Av; Bm; Bv
  
  R = spzeros(I,J)
  for i in Is
      for j in Js[Is .== i]
        R[i,j] = X[i,j] - dot(Um[:,i],Vm[:,j])
      end
  end
  UR = Um #TODO
  VR = Vm #TODO

  for t=1:niter #Iteration
    update_latent!(R,Um,Uv,alpha,tau,Vm,Vv,Am,F)
    R = R'
    update_latent!(R,Vm,Vv, beta,tau,Um,Uv,Bm,G)
    R = R'
   # update_link!((UR, Am, Av, alpha, phiA, F)
   # update_link!(VR, Bm, Bv, beta, phiB, G)
    update_prior!(alpha, phiA, Am, Av, Um, Uv, F)
    update_prior!(beta, phiB, Bm, Bv, Vm, Vv, G)
    if verbose
      print(vecnorm(R),"\n")
    end
  end

 print(Um,"\n")
 print(Vm,"\n")
end
