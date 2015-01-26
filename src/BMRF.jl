include("sampling.jl")
include("purecg.jl")

function BMRF(data::RelationData;
              num_latent::Int = 10,
              lambda_beta     = 1.0,
              alpha           = 5.0,
              burnin   = 500,
              psamples = 100,
              class_cut     = log10(200),
              verbose::Bool = true,
              compute_rhs_change = false)
  correct = Float64[] 

  initModel!(data.entities[1], num_latent, lambda_beta = lambda_beta)
  initModel!(data.entities[2], num_latent, lambda_beta = lambda_beta)

  verbose && println("Sampling")
  err_avg  = 0.0
  roc_avg  = 0.0
  rmse_avg = 0.0

  local probe_rat_all
  rhs_change = ones(burnin + psamples, num_latent)

  ## Gibbs sampling loop
  for i in 1 : burnin + psamples
    time0 = time()

    rel = data.relations[1]

    # Sample from movie hyperparams
    mi = data.entities[2].model

    mi.mu, mi.Lambda = rand( ConditionalNormalWishart(mi.sample, mi.mu0, mi.b0, mi.WI, num_latent) )

    # sampling movie latent vectors
    for mm = 1:data.entities[2].count
      mi.sample[mm, :] = sample_user(mm, rel.data, 2, rel.mean_rating, data.entities[1].model.sample, alpha, mi.mu, mi.Lambda, num_latent)
    end

    # Sample from user hyperparams
    # for BMRF using U - data.F * beta (residual) instead of U
    mi = data.entities[1].model

    # BMRF, instead of mu_u using mu_u + data.F * beta    
    uhat = data.entities[1].F * mi.beta
    mi.mu, mi.Lambda = rand( ConditionalNormalWishart(mi.sample - uhat, mi.mu0, mi.b0, mi.WI, num_latent) )

    # sampling user latent vectors
    for uu = 1:data.entities[1].count
      mi.sample[uu, :] = sample_user(uu, rel.data, 1, rel.mean_rating, data.entities[2].model.sample, alpha, mi.mu + uhat[uu,:]', mi.Lambda, num_latent)
    end

    # sampling beta (using GAMBL-R trick)
    if hasFeatures( data.entities[1] )
      mi.beta, rhs = sample_beta(data.entities[1].F, mi.sample .- mi.mu', mi.Lambda, mi.lambda_beta)
    end

    if compute_rhs_change
      if i > 1
        diff = sqrt(sum( (rhs - rhs_prev) .^ 2, 1 ))
        rhs_change[i,:] = diff ./ sqrt(sum(rhs .^ 2, 1))
      end
      rhs_prev = copy(rhs)
    end

    # clamping maybe needed for MovieLens data
    probe_rat = pred(rel.test_vec, data.entities[2].model.sample, data.entities[1].model.sample, rel.mean_rating)
    #else
    #  probe_rat = pred_clamp(probe_vec, sample_m, sample_u, mean_rating)
    #end

    if i > burnin
      probe_rat_all = (counter_prob*probe_rat_all + probe_rat)/(counter_prob+1)
      counter_prob  = counter_prob + 1
    else
      probe_rat_all = probe_rat
      counter_prob  = 1
    end

    time1    = time()
    correct  = (rel.test_label .== (probe_rat_all .< class_cut) )
    err_avg  = mean(correct)
    err      = mean(rel.test_label .== (probe_rat .< class_cut))
    rmse_avg = sqrt(mean( (rel.test_vec[:,3] - probe_rat_all) .^ 2 ))
    rmse     = sqrt(mean( (rel.test_vec[:,3] - probe_rat) .^ 2 ))
    roc_avg  = AUC_ROC(rel.test_label, -vec(probe_rat_all))
    verbose && @printf("Iteration %d:\t avgAcc %6.4f Acc %6.4f | avgRMSE %6.4f | avgROC %6.4f | FU(%6.2f) FM(%6.2f) Fb(%6.2f) [%2.0fs]\n", i, err_avg, err, rmse_avg, roc_avg, vecnorm(data.entities[1].model.sample), vecnorm(data.entities[2].model.sample), vecnorm(data.entities[1].model.beta), time1 - time0)
  end

  result = Dict{String,Any}()
  result["num_latent"]  = num_latent
  result["RMSE"]        = rmse_avg
  result["accuracy"]    = err_avg
  result["ROC"]         = roc_avg
  result["predictions"] = probe_rat_all
  if compute_rhs_change
    result["rhs_change"] = rhs_change
  end
  return result
end