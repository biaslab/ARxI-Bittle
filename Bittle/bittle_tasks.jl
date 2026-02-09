module BittleTasks
"""
tasks.jl

Task-based communication module for Bittle robot control.
Handles encoding and sending of robot commands as tasks.
"""

using Logging
using SerialPorts
include("bittle_serials.jl"); 
using .BittleSerials: read_line, BittleSerial, SerialPort, list_available_ports, send_data

export send_task, send_task_parallel, serial_write_num_to_byte, serial_write_byte,
        split_task_for_large_angles, encode_string, parse_token_v

const DELAY_BETWEEN_SLICE = 0.001

"""
    encode_string(s::String, encoding::String="utf-8")::Vector{UInt8}

    Encode a string to bytes.

    # Arguments
    - `s::String`: String to encode
    - `encoding::String`: Encoding type (default: "utf-8")

    # Returns
    - `Vector{UInt8}`: Encoded bytes
"""
function encode_string(s::String, encoding::String="utf-8")::Vector{UInt8}
    return codeunits(s)
end

"""
    serial_write_num_to_byte(port, token::Char, var::Vector=[];
                            skill_header::Int=4, frame_size::Int=16)

    Encode numeric data and send via serial port.
    Used for tokens: 'c', 'm', 'u', 'b', 'i', 'K', 'L', etc.

    # Arguments
    - `port`: SerialPort object
    - `token::Char`: Command token
    - `var::Vector`: Variables/angles to send
    - `skill_header::Int`: Header size for skill data
    - `frame_size::Int`: Frame size for animation data
"""
function serial_write_num_to_byte(port, token::Char, var::Vector=[];
                                    skill_header::Int=4, frame_size::Int=16)
    logger = Logging.current_logger()
    @debug "serial_write_num_to_byte: token=$token, var=$var"
    
    in_str = ""
    
    if token == 'K'
        # Skill data encoding
        period = var[1]
        
        if period > 0
            skill_header = 4
        else
            skill_header = 7
        end
    
        if period > 1
            frameSize = 8  # gait
        elseif period == 1
            frameSize = 16  # posture
        else
            frameSize = 20  # behavior
        end
        
        # Divide large angles by 2 if necessary
        angle_ratio = 1
        for row in 1:abs(period)
            for angle in var[skill_header + (row-1)*frameSize + 1:min(skill_header + (row-1)*frameSize + 16, skill_header + (row-1)*frameSize + frameSize)]
                if angle > 125 || angle < -125
                    angle_ratio = 2
                    break
                end
            end
            if angle_ratio == 2
                break
            end
        end
        
        if angle_ratio == 2
            var[4] = 2
            for row in 1:abs(period)
                for i in skill_header + (row-1)*frameSize + 1:min(skill_header + (row-1)*frameSize + 16,skill_header + (row-1)*frameSize + frameSize)
                    var[i] = div(var[i], 2)
                end
            end
            @debug "rescaled: $var"
        end
        
        # Encode as binary
        var_int = [Int8(v) for v in var]
        in_str = string(token) * String(reinterpret(UInt8, var_int)) * "~"
    
    elseif isuppercase(token)
        # Binary encoding for uppercase tokens
        if length(var) > 0
            message = [Int(v) for v in var]
            
            if token == 'B'
                # Adjust timing values
                for l in 1:div(length(message), 2)
                    message[l*2] *= 8
                end
            end
            
            if token == 'W' || token == 'C'
                # Unsigned byte packing
                in_str = string(token) * String(UInt8.(message)) * "~"
            else
                # Signed byte packing
                in_str = string(token) * String(reinterpret(UInt8, Int8.(message))) * "~"
            end
        else
            in_str = string(token) * "~"
        end
    else
        # Space-separated decimal encoding
        message = ""
        for element in var
            message *= "$(round(Int, element)) "
        end
        in_str = string(token) * message * "\n"
        
    end
    
    # Send in chunks with delay
    slice_pos = 1
    while slice_pos <= length(in_str)
        chunk_end = min(slice_pos + 19, length(in_str))
        chunk = in_str[slice_pos:chunk_end]
        
        # send_data(port, chunk)
        cchunk = codeunits(chunk)
        SerialPorts.write(port.serial, cchunk)
        
        slice_pos = chunk_end + 1
        sleep(DELAY_BETWEEN_SLICE)

        @debug "Sent: $cchunk"
    end
end

"""
    serial_write_byte(port, var::Vector)

Send a command string directly via serial port.

# Arguments
- `port`: SerialPort object
- `var::Vector`: Command tokens/strings to send
"""
function serial_write_byte(port, var::Vector)
    logger = Logging.current_logger()
    @debug "serial_write_byte: var=$var"
    
    if length(var) == 0
        return nothing
    end
    
    token = var[1]
    in_str = ""
    
    if token in ['c', 'm', 'i', 'b', 'u', 't'] && length(var) >= 2
        # Space-separated command
        for element in var
            in_str *= string(element) * " "
        end
        in_str *= "\n"
        
    elseif token == 'L' || token == 'I'
        # Binary angle encoding
        var_copy = copy(var)
        if length(var_copy[1]) > 1
            insert!(var_copy, 2, var_copy[1][2:end])
        end
        
        # Convert to integers (skip first element which is token)
        values = [Int8(v) for v in var_copy[2:end]]
        in_str = string(token) * String(reinterpret(UInt8, values)) * "~"
        
    elseif token in ['w', 'k', 'X', 'g']
        # String command with newline
        in_str = var[1] * "\n"
    else
        # Single token command
        in_str = token * "\n"
    end
    
    @debug "Sending: $in_str"
    SerialPorts.write(port.serial, codeunits(in_str))
    sleep(0.01)
end

"""
    print_serial_message(port, token::Char; threshold::Int=3, timeout::Float64=0.0)

Read and print serial messages until expected token received.

# Arguments
- `port`: SerialPort object
- `token::Char`: Expected response token
- `threshold::Int`: Timeout threshold in seconds
- `timeout::Float64`: Maximum overall timeout (0 = no limit)

# Returns
- `String`: response or -1 on timeout
"""
function print_serial_message(port, token::Char; threshold::Int=3, timeout::Float64=0.0)
    if token == 'k' || token == 'K'
        threshold = 8
    else
        threshold = 3
    end
    
    if contains(string(token), "X")
        token = 'X'
    end
    
    start_time = time()
    all_prints = ""
    
    if port !== nothing && port.is_open && port.serial !== nothing
    
        response = BittleSerials.read_line(port.serial)
        @debug "Response: $response"
        return response
    end
    
    now = time()
    if (now - start_time) > threshold
        @debug "Elapsed time: $threshold seconds"
        threshold += 2
        
        if threshold > 5
            return -1
        end
    end
    
    if timeout > 0 && (now - start_time) > timeout
        return -1
    end
end

"""
    split_task_for_large_angles(task::Vector)

Split tasks containing large angles into multiple tasks.

# Arguments
- `task::Vector`: Task in format [token, angles/vars, delay]

# Returns
- `Vector{Vector}`: List of tasks
"""
function split_task_for_large_angles(task::Vector)::Vector{Vector}
    token = task[1]
    queue = []
    
    if length(task) > 2 && (token == 'L' || token == 'I')
        var = copy(task[2])
        indexed_list = []
        
        if token == 'L'
            # Process 4x4 matrix
            for i in 1:4
                for j in 1:4
                    angle = var[4*j + i]
                    if angle < -125 || angle > 125
                        push!(indexed_list, 4*j + i)
                        push!(indexed_list, angle)
                        var[4*j + i] = clamp(angle, -125, 125)
                    end
                end
            end
            
            if length(var) > 0
                push!(queue, ['L', var, task[end]])
            end
            
            if length(indexed_list) > 0
                queue[1][end] = 0.01
                push!(queue, ['i', indexed_list, task[end]])
            end
            
        elseif token == 'I'
            if minimum(var) < -125 || maximum(var) > 125
                task[1] = 'i'
            end
            push!(queue, task)
        end
    else
        push!(queue, task)
    end
    
    return queue
end

"""
    send_task(port_list::Dict, port, task::Vector; timeout::Float64=0.0)

Send a single task to the robot via serial port.

# Arguments
- `port_list::Dict`: Dictionary of active ports
- `port`: SerialPort object
- `task::Vector`: Task [token, variables, delay]
- `timeout::Float64`: Read timeout

# Returns
- `Union{Tuple, Int}`: Message response or -1 on error
"""
function send_task(port_list::Dict, port, task::Vector; timeout::Float64=0.1)
    logger = Logging.current_logger()
    @debug "send_task: $task"
    
    last_message = -1
    
    if port !== nothing
        try
            if length(task) == 2
                serial_write_byte(port, [task[1]])
            elseif isa(task[2][1], Number)
                serial_write_num_to_byte(port, task[1], task[2])
            else
                serial_write_byte(port, task[2])
            end
            sleep(timeout)
            last_message = print_serial_message(port, task[1], timeout=timeout)
            
        catch e
            @error "Error sending task: $e" exception=(e,catch_backtrace())
            if port in keys(port_list)
                delete!(port_list, port)
            end
            last_message = -1
        end
    else
        last_message = -1
    end
    
    return last_message
end

"""
    send_task_parallel(port_list::Dict, ports::Vector, task::Vector; timeout::Float64=0.0)

Send a task to multiple ports in parallel using threading.

# Arguments
- `port_list::Dict`: Dictionary of active ports
- `ports::Vector`: List of SerialPort objects
- `task::Vector`: Task to send
- `timeout::Float64`: Read timeout

# Returns
- `Union{Tuple, Int}`: Last message response
"""
function send_task_parallel(port_list::Dict, ports::Vector, task::Vector; timeout::Float64=0.0)
    last_message = -1
    
    # Create tasks for each port
    tasks = []
    for p in ports
        t = Threads.@spawn send_task(port_list, p, task, timeout=timeout)
        push!(tasks, t)
    end
    
    # Wait for all tasks to complete
    for t in tasks
        try
            result = fetch(t)
            last_message = result
        catch e
            @error "Error in parallel task: $e"
        end
    end
    
    return last_message
end


function parse_token_v(response::String)
    response_splitted = split(response, "\n")
    response_longest = argmax(length.(response_splitted))
    response_stripped = strip(response_splitted[response_longest], '\r')
    response_splitted = split(response_stripped, "\t")
    response_splitted = response_splitted[response_splitted .!= ""]
    return parse.(Float64, response_splitted)
end

end