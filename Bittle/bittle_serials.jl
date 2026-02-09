module BittleSerials

"""
    serial.jl
    
Serial communication module for Bittle robot control.
Provides serial port management and communication capabilities.
"""

using SerialPorts
using Serialization
using Sockets
using Logging

export BittleSerial, SerialPort, open_port, close_port, read_size, read_line, send_data, 
       list_available_ports, get_port_info, print_port_info,
       send_task, send_task_parallel, serial_write_num_to_byte, serial_write_byte,
       print_serial_message, split_task_for_large_angles,
       bytes_to_hex, hex_to_bytes, pack_signed_bytes, pack_unsigned_bytes,
       unpack_signed_bytes, unpack_unsigned_bytes, find_ports_by_pattern,
       get_platform_serial_paths

# Global state
const GLOBAL_PORT_STATE = Dict{String, Bool}()
    const DELAY_BETWEEN_SLICE = 0.001

"""
    BittleSerial

Wrapper for SerialPorts.SerialPort with additional metadata for Bittle robot control.

# Fields
- `port::String`: Port name (e.g., "/dev/ttyUSB0")
- `bps::Int`: Baud rate (bits per second)
- `timeout::Float64`: Read timeout in seconds
- `serial::Union{Nothing, SerialPorts.SerialPort}`: SerialPorts SerialPort object
- `is_open::Bool`: Whether the port is currently open
- `data::Union{Nothing, Vector{UInt8}}`: Last received data
"""
mutable struct BittleSerial
    port::String
    bps::Int
    timeout::Float64
    serial::Union{Nothing, SerialPorts.SerialPort}
    is_open::Bool
    data::Union{Nothing, Vector{UInt8}}
    function BittleSerial(port::String, bps::Int, timeout::Float64)
        new(port, bps, timeout, nothing, false, nothing)
    end
end

"""
    list_available_ports()::Vector{String}

List all available serial ports on the system.

# Returns
- `Vector{String}`: List of port names (e.g., ["COM3", "COM4"])
"""
function list_available_ports()::Vector{String}
    ports = String[]
    for file in readdir("/dev")
        if startswith(file, "ttyUSB") || startswith(file, "ttyACM")
            push!(ports, "/dev/$file")
        end
    end
    return ports
end

"""
    open_port(bport::BittleSerial)::Bool

Open a serial port connection.

# Arguments
- `bport::BittleSerial`: BittleSerial object to open

# Returns
- `Bool`: True if successfully opened, false otherwise
"""
function open_port(bport::BittleSerial)::Bool
    try
        bport.serial = SerialPorts.SerialPort(bport.port, bport.bps)
        bport.is_open = true
        GLOBAL_PORT_STATE[bport.port] = true
        return true
    catch e
        @error "Failed to open port $(bport.port): $e"
        return false
    end
end

"""
    close_port(bport::BittleSerial)::Bool

Close a serial port connection.

# Arguments
- `bport::BittleSerial`: BittleSerial object to close

# Returns
- `Bool`: True if successfully closed, false otherwise
"""
function close_port(bport::BittleSerial)::Bool
    try
        if bport.is_open && bport.serial !== nothing
            close(bport.serial)
            bport.is_open = false
            bport.serial = nothing
            GLOBAL_PORT_STATE[bport.port] = false
            return true
        end
        return false
    catch e
        @error "Failed to close port $(bport.port): $e"
        return false
    end
end

"""
    read_size(bport::BittleSerial, size::Int)::Vector{UInt8}

Read a specified number of bytes from the serial port.

# Arguments
- `bport::BittleSerial`: BittleSerial object to read from
- `size::Int`: Number of bytes to read

# Returns
- `Vector{UInt8}`: Bytes read from the port
"""
function read_size(bport::BittleSerial, size::Int)::Vector{UInt8}
    try
        if bport.is_open && bport.serial !== nothing
            data = read(bport.serial, size)
            bport.data = data
            return data
        end
        return UInt8[]
    catch e
        @error "Error reading from port: $e"
        return UInt8[]
    end
end

"""
    read_line(port::SerialPort)::String

Read a line from the serial port until newline character.

# Arguments
- `port::SerialPort`: Serial object to read from

# Returns
- `String`: Line read from the port (without newline)
"""
function read_line(port::SerialPort)
    try
        if port !== nothing
            readbytes = SerialPorts.readavailable(port)
            @debug "Read bytes: $readbytes"
            # return error(42)
            return readbytes  
        end
    catch e
        @error "Error reading line from port: $e" exception=(e,catch_backtrace())
        return ""
    end
end

"""
    send_data(bport::BittleSerial, data::Union{String, Vector{UInt8}})

Send data to the serial port.

# Arguments
- `bport::BittleSerial`: BittleSerial object to write to
- `data::Union{String, Vector{UInt8}}`: Data to send
"""
function send_data(bport::BittleSerial, data::Union{String, Vector{UInt8}})
    try
        if bport.is_open && bport.serial !== nothing
            if isa(data, String)
                write(bport.serial, codeunits(data))
            else
                write(bport.serial, data)
            end
        else
            @warn "Port not open: $(bport.port)"
        end
    catch e
        @error "Error sending data: $e"
    end
end

"""
    get_port_info(bport::BittleSerial)::Dict{String, Any}

Get information about a serial port.

# Arguments
- `bport::BittleSerial`: BittleSerial object

# Returns
- `Dict{String, Any}`: Port information
"""
function get_port_info(bport::BittleSerial)::Dict{String, Any}
    return Dict(
        "port" => bport.port,
        "bps" => bport.bps,
        "timeout" => bport.timeout,
        "is_open" => bport.is_open
    )
end

"""
    print_port_info(bport::BittleSerial)

Print detailed information about a serial port.

# Arguments
- `bport::BittleSerial`: BittleSerial object
"""
function print_port_info(bport::BittleSerial)
    info = get_port_info(bport)
    println("Port Information:")
    println("  Port: $(info["port"])")
    println("  Baud Rate: $(info["bps"]) bps")
    println("  Timeout: $(info["timeout"]) s")
    println("  Is Open: $(info["is_open"])")
end

"""
    bytes_to_hex(data::Vector{UInt8})::String

Convert bytes to hexadecimal string representation.

# Arguments
- `data::Vector{UInt8}`: Bytes to convert

# Returns
- `String`: Hexadecimal string (e.g., "0xaabbccdd")
"""
function bytes_to_hex(data::Vector{UInt8})::String
    return "0x" * bytes2hex(data)
end

"""
    hex_to_bytes(hex_str::String)::Vector{UInt8}

Convert hexadecimal string to bytes.

# Arguments
- `hex_str::String`: Hexadecimal string (with or without 0x prefix)

# Returns
- `Vector{UInt8}`: Bytes represented by the hex string
"""
function hex_to_bytes(hex_str::String)::Vector{UInt8}
    # Remove 0x prefix if present
    hex_clean = startswith(hex_str, "0x") ? hex_str[3:end] : hex_str
    
    if isodd(length(hex_clean))
        hex_clean = "0" * hex_clean
    end
    
    return hex2bytes(hex_clean)
end

"""
    pack_signed_bytes(values::Vector)::Vector{UInt8}

Pack signed integers into bytes.

# Arguments
- `values::Vector`: Signed integer values

# Returns
- `Vector{UInt8}`: Packed bytes
"""
function pack_signed_bytes(values::Vector)::Vector{UInt8}
    signed_values = [Int8(v) for v in values]
    return reinterpret(UInt8, signed_values)
end

"""
    pack_unsigned_bytes(values::Vector)::Vector{UInt8}

Pack unsigned integers into bytes.

# Arguments
- `values::Vector`: Unsigned integer values

# Returns
- `Vector{UInt8}`: Packed bytes
"""
function pack_unsigned_bytes(values::Vector)::Vector{UInt8}
    return UInt8.(values)
end

"""
    unpack_signed_bytes(data::Vector{UInt8})::Vector{Int8}

Unpack bytes into signed integers.

# Arguments
- `data::Vector{UInt8}`: Bytes to unpack

# Returns
- `Vector{Int8}`: Signed integer values
"""
function unpack_signed_bytes(data::Vector{UInt8})::Vector{Int8}
    return reinterpret(Int8, data)
end

"""
    unpack_unsigned_bytes(data::Vector{UInt8})::Vector{UInt8}

Unpack bytes into unsigned integers (identity operation).

# Arguments
- `data::Vector{UInt8}`: Bytes to unpack

# Returns
- `Vector{UInt8}`: Unsigned integer values
"""
function unpack_unsigned_bytes(data::Vector{UInt8})::Vector{UInt8}
    return data
end

"""
    get_platform_serial_paths()::Vector{String}

Get platform-specific serial device paths to check.

# Returns
- `Vector{String}`: List of possible serial device paths
"""
function get_platform_serial_paths()::Vector{String}
    if Sys.iswindows()
        # Windows COM ports (COM1 through COM20)
        return ["COM$i" for i in 1:20]
    elseif Sys.islinux()
        # Linux device paths
        linux_paths = []
        for file in readdir("/dev/", join=true)
            name = basename(file)
            if startswith(name, "ttyUSB") || startswith(name, "ttyACM") || startswith(name, "ttyS")
                push!(linux_paths, file)
            end
        end
        return linux_paths
    elseif Sys.isapple()
        # macOS device paths
        macos_paths = []
        if isdir("/dev")
            for file in readdir("/dev/", join=true)
                name = basename(file)
                if startswith(name, "tty.") || startswith(name, "cu.")
                    push!(macos_paths, file)
                end
            end
        end
        return macos_paths
    else
        return []
    end
end

"""
    find_ports_by_pattern(pattern::String)::Vector{String}

Find serial ports matching a specific pattern.

# Arguments
- `pattern::String`: Pattern to match (e.g., "COM" or "ttyUSB")

# Returns
- `Vector{String}`: Matching port names
"""
function find_ports_by_pattern(pattern::String)::Vector{String}
    all_ports = get_platform_serial_paths()
    return filter(p -> contains(lowercase(p), lowercase(pattern)), all_ports)
end

"""
    encode_uint8_array(values::Vector)::String

Encode a vector of integers as a string for transmission.

# Arguments
- `values::Vector`: Integer values to encode

# Returns
- `String`: Encoded string representation
"""
function encode_uint8_array(values::Vector)::String
    return String(reinterpret(UInt8, [Int8(v) for v in values]))
end

"""
    encode_space_separated(values::Vector)::String

Encode values as space-separated decimal numbers.

# Arguments
- `values::Vector`: Values to encode

# Returns
- `String`: Space-separated string
"""
function encode_space_separated(values::Vector)::String
    parts = [string(Int(round(v))) for v in values]
    return join(parts, " ") * " "
end

"""
    parse_response(response::String)::Dict{String, Any}

Parse a response message from the robot.

# Arguments
- `response::String`: Response message

# Returns
- `Dict{String, Any}`: Parsed response data
"""
function parse_response(response::String)::Dict{String, Any}
    # Remove whitespace and newlines
    response_clean = strip(response)
    
    # Try to split by various delimiters
    parts = split(response_clean, r"[\s,;]", keepempty=false)
    
    # Attempt to convert to numbers
    values = Float64[]
    for part in parts
        try
            push!(values, parse(Float64, part))
        catch
            # Non-numeric value, keep as string
        end
    end
    
    return Dict(
        "raw" => response,
        "clean" => response_clean,
        "parts" => parts,
        "values" => values
    )
end

"""
    clamp_angle(angle::Real; min_val::Int=-125, max_val::Int=125)::Int

Clamp an angle value to valid servo range.

# Arguments
- `angle::Real`: Angle value to clamp
- `min_val::Int`: Minimum angle (default: -125)
- `max_val::Int`: Maximum angle (default: 125)

# Returns
- `Int`: Clamped angle value
"""
function clamp_angle(angle::Real; min_val::Int=-125, max_val::Int=125)::Int
    return clamp(Int(round(angle)), min_val, max_val)
end

"""
    scale_angles(angles::Vector, scale_factor::Real)::Vector{Int}

Scale a vector of angles by a factor.

# Arguments
- `angles::Vector`: Angle values to scale
- `scale_factor::Real`: Factor to scale by

# Returns
- `Vector{Int}`: Scaled angle values
"""
function scale_angles(angles::Vector, scale_factor::Real)::Vector{Int}
    return [Int(round(a / scale_factor)) for a in angles]
end

"""
    create_skill_frame(servo_angles::Vector; frame_duration::Int=32, 
                       mirror_legs::Bool=false)::Vector{Int}

Create a single frame of animation data for robot movement.

# Arguments
- `servo_angles::Vector`: Angles for up to 16 servos
- `frame_duration::Int`: Frame duration in milliseconds
- `mirror_legs::Bool`: Whether to mirror left/right legs

# Returns
- `Vector{Int}`: Frame data with header
"""
function create_skill_frame(servo_angles::Vector; frame_duration::Int=32, 
                           mirror_legs::Bool=false)::Vector{Int}
    # Frame format: [16 servo angles, duration (4 bytes), unknown (2 bytes)]
    frame = zeros(Int, 20)
    
    # Limit to 16 servo values
    for i in 1:min(16, length(servo_angles))
        frame[i] = clamp_angle(servo_angles[i])
    end
    
    # Add frame duration (lower 2 bytes)
    frame[17] = frame_duration & 0xFF
    frame[18] = (frame_duration >> 8) & 0xFF
    
    # Remaining bytes for future use
    frame[19] = 0
    frame[20] = 0
    
    return frame
end

end # module
