using Revise
using SerialPorts
using Logging

includet("../Bittle/bittle_serials.jl"); using .BittleSerials
includet("../Bittle/bittle_tasks.jl"); using .BittleTasks

logger = ConsoleLogger(stdout, Logging.Debug)
global_logger(logger)



ports = list_available_ports()
bport = BittleSerial(ports[1],115200,0.0)
if open_port(bport)
    println("Connected")
end

# Send joint angle command
# task = ['d', [0, -30, 20, 45], 0.2]
task = ['I', [8, 0, 12, -20, 9, 0, 13, 0, 11, 0, 15, 0, 10, 0, 14, 0], 0.0]
good_ports = Dict(bport => true)
send_task(good_ports, bport, task)

# Read out IMU
response = send_task(good_ports, bport, ['v', 0])
IMUstate = parse_token_v(response)

# Close port
println("\nClosing connection...")
close_port(bport)
@info "Connection closed"