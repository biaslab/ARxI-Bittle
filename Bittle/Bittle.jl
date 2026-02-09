module Bittle

include("bittle_serials.jl")
include("bittle_tasks.jl")
include("IMU.jl")

using .BittleSerials
using .BittleTasks
using .IMU

export BittleSerial, send_task, open_port, close_port, print_serial_message, list_available_ports, acc2pos,proj2psd, parse_token_v

end