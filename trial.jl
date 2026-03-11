using Revise
using LinearAlgebra
using Distributions
using Dates
using Random
using JLD2
using Printf
using Logging

includet("ARxI/ARxI.jl"); using .ARxI
# includet("Bittle/Bittle.jl"); using .Bittle


# Time
len_trial = 10
len_horizon = 3
now = Dates.format(Dates.now(), "yyyy-mm-dd-HH-MM-SS")

# Dimensionalities
order_y = 2
order_u = 2
Dy = 6
Du = 8
Dx = order_y*Dy + (order_u+1)*Du

# Prior parameters
ν0 = 100.0
Ω0 = 1e-1 * Diagonal(ones(Dy))
Λ0 = 1e-4 * Diagonal(ones(Dx))
M0 = 1e-12 * randn(Dx, Dy)
Υ = 1e-12 * Diagonal(ones(Du))

# Setpoint (desired observation)
m_star = [0.0, -10.0, 0.0, 10.0, 0.0, 0.0]
v_star = [1e0, 1e-5, 1e0, 1e-5, 1e3, 1e3]
S_star = Diagonal(v_star)
goal = MvNormal(m_star, S_star)

# Control limits
u_lims = (-20, 20)

# Preallocate
times = zeros(len_trial)
y_ = zeros(Dy, len_trial)
u_ = zeros(Du, len_trial)
Ms = zeros(Dx, Dy, len_trial)
Λs = zeros(Dx, Dx, len_trial)
Ωs = zeros(Dy, Dy, len_trial)
νs = zeros(len_trial)

Ms[:, :, 1] = M0
Λs[:, :, 1] = Λ0
Ωs[:, :, 1] = Ω0
νs[1] = ν0

yb = zeros(Dy,order_y)
ub = zeros(Du,order_u+1)

IMUstate = MvNormal(zeros(9), I(9))

port = PetoiBittle.find_bittle_port()
@info "Using port $(port) to connect to PetoiBittle"
connection = PetoiBittle.connect(port)
@info "Sleeping for 5 seconds to let the Petoi Bittle initialize"
sleep(5)

task = PetoiBittle.MoveJoints(
    (id = 8,  angle = 0),
    (id = 12, angle = 0),
    (id = 9,  angle = 0),
    (id = 13, angle = 0),
    (id = 11, angle = 0),
    (id = 15, angle = 0),
    (id = 10, angle = 0),
    (id = 14, angle = 0)
)
PetoiBittle.send_command(connection, task)

# ports = Bittle.list_available_ports()
# bport = Bittle.BittleSerial(ports[1],115200,0.0)
# if Bittle.open_port(bport)
#     println("Connected")
# end
# good_ports = Dict(bport => true)
# Bittle.send_task(good_ports, bport, ['I', [8, 0, 12, 0, 9, 0, 13, 0, 11, 0, 15, 0, 10, 0, 14, 0], 0.0])
# response = Bittle.send_task(good_ports, bport, ['v', 0])

try
    
    times[1] = time()
    for k in 2:len_trial
    # k = 2
        times[k] = time()
        dt = times[k] - times[k-1]
        @info "tstep = $k/$len_trial, time = $(times[k])"

        # Interact with environment
        task = PetoiBittle.MoveJoints(
            (id= 8, angle=Int(round(u_[1, k-1]))),
            (id=12, angle=Int(round(u_[2, k-1]))),
            (id= 9, angle=Int(round(u_[3, k-1]))),
            (id=13, angle=Int(round(u_[4, k-1]))),
            (id=11, angle=Int(round(u_[5, k-1]))),
            (id=15, angle=Int(round(u_[6, k-1]))),
            (id=10, angle=Int(round(u_[7, k-1]))),
            (id=14, angle=Int(round(u_[8, k-1]))),)
        PetoiBittle.send_command(connection, task)

        # Read IMU (placeholder)
        ypracc = PetoiBittle.send_command(connection, PetoiBittle.GyroStats())
        # response = Bittle.send_task(good_ports, bport, ['v', 0])
        # ypracc = Bittle.parse_token_v(response)
        # # ypracc = parse.(Float64, split(ypracc[2])[end-5:end])
        dt = min(1.0, dt)
        IMUstate = acc2pos(ypracc[end-2:end] / 1e3, IMUstate, dt=dt, reg=dt)
        y_[:, k] = vcat(ypracc[1:3], IMUstate.μ[1:3])

        # Update parameters
        @info "Updating parameters.."
        params = infer_params(y_[:, k],
                              yb, 
                              ub,
                              Ms[:, :, k-1], 
                              Λs[:, :, k-1], 
                              Ωs[:, :, k-1], 
                              νs[k-1])

        Ms[:, :, k] = params[1]
        Λs[:, :, k] = params[2]
        Ωs[:, :, k] = params[3]
        νs[k] = params[4]

        # Update buffer
        yb = backshift(yb, y_[:, k])

        # Plan actions
        @info "Planning.."
        policy = infer_actions(yb, 
                               ub,
                               Ms[:, :, k], 
                               Λs[:, :, k], 
                               Ωs[:, :, k], 
                               νs[k],
                               Υ, 
                               m_star, 
                               S_star, 
                               len_horizon)

        u_[:, k] = clamp.(round.(policy[1]), u_lims[1], u_lims[2])
        @info u_[:, k]

        # Update buffer
        ub = backshift(ub, u_[:, k])
    end

    PetoiBittle.disconnect(connection)
    @info "Ports closed."

    println("Saving results..")
    JLD2.@save "results/trials/agent-ARxI-$now-times.jls" times
    JLD2.@save "results/trials/agent-ARxI-$now-actions.jls" u_
    JLD2.@save "results/trials/agent-ARxI-$now-observations.jls" y_
    println("Experiment completed.")

catch e
    @info "Exception"
    @info e
    @info catch_backtrace()         
    PetoiBittle.disconnect(connection)
    exit(0)
end
