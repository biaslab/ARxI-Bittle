module IMU

using LinearAlgebra
using Distributions

export proj2psd, acc2pos


function proj2psd(S::AbstractMatrix; epsilon=1e-8)
    L,V = eigen(S)
    S = V*diagm(max.(epsilon,real(L)))*V'
    return (S+S')/2
end

function acc2pos(acc::Vector{Float64}, prev_state; dt=1.0, reg=1e-3)
    """
    Accelerations -> positions
    
    This uses a Kalman filter for accelerometer integration.
    """

    A = [1 0 0 dt 0 0 dt^2/2 0 0;
         0 1 0 0 dt 0 0 dt^2/2 0;
         0 0 1 0 0 dt 0 0 dt^2/2;
         0 0 0 1 0 0 dt 0 0;
         0 0 0 0 1 0 0 dt 0;
         0 0 0 0 0 1 0 0 dt;
         0 0 0 0 0 0 1 0 0;
         0 0 0 0 0 0 0 1 0;
         0 0 0 0 0 0 0 0 1]
    σ = 1.0
    block1 = Diagonal(fill(dt^5/20, 3))
    block2 = Diagonal(fill(dt^4/8, 3))
    block3 = Diagonal(fill(dt^3/6, 3))
    block4 = Diagonal(fill(dt^2/2, 3))
    block5 = Diagonal(fill(dt, 3))
    Q = σ * [block1 block2 block3;
             block2 block3 block4;
             block3 block4 block5]
    C = [0 0 0 0 0 0 1 0 0;
         0 0 0 0 0 0 0 1 0;
         0 0 0 0 0 0 0 0 1]
    ρ = 1.0
    R = Diagonal(fill(ρ, 3))
    
    # Kalman predictions
    state_pred_m = A * prev_state.μ
    state_pred_S = A * prev_state.Σ * A' + Q
    
    # Kalman update
    Is = C * state_pred_S * C' + R
    Kg = state_pred_S * C' * inv(Is)
    state_m = state_pred_m + Kg * (acc - C * state_pred_m)
    state_S = (I(9) - Kg * C) * state_pred_S * (I(9) - Kg * C)' + Kg * R * Kg' + reg * I(9)
    
    # Ensure numerical stability by projecting to PSD
    state_S = proj2psd(state_S, epsilon=1.0e-6)
    
    return MvNormal(state_m, state_S)
end

end