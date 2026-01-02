using Revise
using Logging; Logging.LogLevel(-1000)
using LinearAlgebra
using Distributions
using Optim
using ForwardDiff
using RxInfer

includet("src/util.jl");
includet("distributions/matrix_normal_wishart.jl");
includet("distributions/unboltzmann.jl");
includet("distributions/mv_location_scale_t.jl");
includet("nodes/MARX.jl");
includet("nodes/matrix_normal_wishart.jl");
includet("rules/MARX/in.jl");
includet("rules/MARX/out.jl");
includet("rules/MARX/parameter.jl");
includet("rules/matrix_normal_wishart/out.jl")



@model function learning(y_k,y_kmin1,y_kmin2,u_k,u_kmin1,u_kmin2, M_kmin1,Λ_kmin1,Ω_kmin1,ν_kmin1)

    # Prior distribution over MARX parameters
    Θ ~ MatrixNormalWishart(M_kmin1, Λ_kmin1, Ω_kmin1, ν_kmin1)

    # MARX Likelihood
    y_k ~ MARX(y_kmin1,y_kmin2,u_k,u_kmin1,u_kmin2,Θ)

end

@model function planning(y_tmin1,y_tmin2,u_tmin1,u_tmin2, M_k,Λ_k,Ω_k,ν_k,Υ,m_star,S_star,len_horizon)

    # Posterior distribution over MARX parameters
    Θ   ~ MatrixNormalWishart(M_k,Λ_k,Ω_k,ν_k)

    # Action priors
    u_[1] ~ MvNormalMeanPrecision(zeros(Du),Υ)
    u_[2] ~ MvNormalMeanPrecision(zeros(Du),Υ)

    # MARX likelihood for t = 1,2
    y_[1] ~ MARX(y_tmin1,y_tmin2,u_[1],u_tmin1,u_tmin2,Θ)
    y_[2] ~ MARX(y_[1],y_tmin1,u_[2],u_[1],u_tmin1,Θ)

    for t = 3:len_horizon

        u_[t] ~ MvNormalMeanPrecision(zeros(Du),Υ)
        y_[t] ~ MARX(y_[t-1],y_[t-2],u_[t],u_[t-1],u_[t-2],Θ)

    end
    
    # Goal prior at final horizon point
    y_[len_horizon] ~ MvNormalMeanCovariance(m_star,S_star)
end

function posterior_predictive(x_t,M,Λ,Ω,ν,Dx,Dy) 
    "Posterior predictive given parameter beliefs and MARX buffer"
    return ( ν-Dy+1, M'*x_t, 1/(ν-Dy+1) * Ω * (1 + x_t'*inv(Λ)*x_t) )
end

function logevidence(y,x,M,Λ,Ω,ν,Dx,Dy)
    "Log evidence of MARX model given parameter beliefs, MARX buffer and current output"
    η, μ, Σ = posterior_predictive(x,M,Λ,Ω,ν,Dx,Dy)
    return -1/2*(Dy*log(η*π) +logdet(Σ) - 2*logmultigamma(Dy, (η+Dy)/2) + 2*logmultigamma(Dy, (η+Dy-1)/2) + (η+Dy)*log(1 + 1/η*(y-μ)'*inv(Σ)*(y-μ)) )
end

function mutualinfo(Σ) 
    "Mutual information between posterior predictive and parameter posterior"  
    return 1/2*logdet(Σ)
end

function crossentropy(m_star, S_star, η,μ,Σ)
    "Cross-entropy between posterior predictive and goal prior (constant terms dropped)"  
    return 1/2*( η/(η-2)*tr(inv(S_star)*Σ) + (μ-m_star)'*inv(S_star)*(μ-m_star) ) 
end 

# Parameters
Mu = 2
My = 2
Dy = 6
Du = 8
Dx = My*Dy + (Mu+1)*Du

# Limits of controller
global u_lims = (-20.0, 20.0)

function infer_params(new_y,
                      ybuffer1,
                      ybuffer2,
                      ubuffer1,
                      ubuffer2,
                      ubuffer3,
                      M_kmin1,
                      Λ_kmin1,
                      Ω_kmin1,
                      ν_kmin1)

    Ω_kmin1 = cat(Ω_kmin1...,dims=2)
    Λ_kmin1 = cat(Λ_kmin1...,dims=2)
    M_kmin1 = cat(M_kmin1...,dims=2)
    M_kmin1 = Matrix(M_kmin1')

    res = infer(
        model = learning(M_kmin1 = M_kmin1,
                         Λ_kmin1 = Λ_kmin1,
                         Ω_kmin1 = Ω_kmin1,
                         ν_kmin1 = ν_kmin1,),
        data    = (y_k = new_y,
                   y_kmin1 = ybuffer1,
                   y_kmin2 = ybuffer2,
                   u_k     = ubuffer1,
                   u_kmin1 = ubuffer2,
                   u_kmin2 = ubuffer3,),
        options = (limit_stack_depth = 100,),
    )

    return params(res.posteriors[:Θ])
end

function infer_actions(ybuffer1,
                       ybuffer2, 
                       ubuffer1,
                       ubuffer2,
                       M_k,
                       Λ_k,
                       Ω_k,
                       ν_k,
                       Υ,
                       m_star,
                       S_star, 
                       len_horizon;
                       num_iters=10)

    # Reconstruct matrices passed from Python nested lists
    S_star = cat(S_star...,dims=2)                      
    Υ      = cat(Υ...,dims=2)
    Ω_k    = cat(Ω_k...,dims=2)
    Λ_k    = cat(Λ_k...,dims=2)
    M_k    = cat(M_k...,dims=2)
    M_k    = Matrix(M_k')
    ν_k    = convert(Float64,ν_k)

    inits = @initialization begin
        q(Θ)  = MatrixNormalWishart(M_k,Λ_k,Ω_k,ν_k)
        q(y_) = vague(MvNormalMeanCovariance,Dy)
        q(u_) = vague(MvNormalMeanCovariance,Du)
    end

    cons = @constraints begin
        q(y_,u_,Θ) = q(y_)q(u_)q(Θ)
        q(y_) = q(y_[begin])..q(y_[end])
        q(u_) = q(u_[begin])..q(u_[end])
        q(u_) :: PointMassFormConstraint()
    end

    res = infer(
        model = planning(M_k         = M_k,
                         Λ_k         = Λ_k,
                         Ω_k         = Ω_k,
                         ν_k         = ν_k,
                         Υ           = Υ,
                         m_star      = m_star, 
                         S_star      = S_star,
                         len_horizon = len_horizon,),
        data = (y_tmin1 = ybuffer1,
                y_tmin2 = ybuffer2,
                u_tmin1 = ubuffer1,
                u_tmin2 = ubuffer2,),
        initialization  = inits,
        constraints     = cons,
        iterations      = num_iters, 
        options         = (limit_stack_depth=100,),
        returnvars      = (u_ = KeepLast(),),
    )

    return mode.(res.posteriors[:u_])
end

