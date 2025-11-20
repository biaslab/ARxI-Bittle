#!/usr/bin/python3
#  -*- coding: UTF-8 -*-

# MindPlus
# Python
from OpenCat.OpenCatPythonAPI.PetoiRobot import *
import datetime as dtime
import numpy as np
import numpy.random as rnd
import timeit
from scipy.stats import multivariate_normal
from scipy.linalg import inv,det
from scipy.optimize import minimize

from julia import Julia
jl = Julia(compiled_modules=False)
jl.eval("include(\"ARxI/live.jl\")")

def acc2pos(acc, prev_state, dt=1.0, reg=1e-3):
    "Kalman filter for accelerometer integration"

    # State transition matrix
    A = np.array([[1, 0, 0, dt,  0,  0, dt**2/2,       0,       0],
                  [0, 1, 0,  0, dt,  0,       0, dt**2/2,       0],
                  [0, 0, 1,  0,  0, dt,       0,       0, dt**2/2],
                  [0, 0, 0,  1,  0,  0,      dt,       0,       0],
                  [0, 0, 0,  0,  1,  0,       0,      dt,       0],
                  [0, 0, 0,  0,  0,  1,       0,       0,      dt],
                  [0, 0, 0,  0,  0,  0,       1,       0,       0],
                  [0, 0, 0,  0,  0,  0,       0,       1,       0],
                  [0, 0, 0,  0,  0,  0,       0,       0,       1]])
    
    # Process noise covariance matrix
    σ   = 1.0   
    block1 = np.diag(np.repeat([dt**5/20], 3))
    block2 = np.diag(np.repeat([dt**4/8], 3))
    block3 = np.diag(np.repeat([dt**3/6], 3))
    block4 = np.diag(np.repeat([dt**2/2], 3))
    block5 = np.diag(np.repeat([dt], 3))
    Q = σ*np.block([[block1,block2,block3],
                    [block2,block3,block4],
                    [block3,block4,block5]])
    
    # Measurement matrix
    C = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1]])
    
    # Measurement noise covariance matrix
    ρ = 1.0
    R = np.diag(ρ*np.ones(3))

    # Prediction step
    state_pred_m = A @ prev_state.mean
    state_pred_S = A @ prev_state.cov @ A.T + Q

    # Correction step
    Is      = C @ state_pred_S @ C.T + R
    Kg      = state_pred_S @ C.T @ inv(Is)
    state_m = state_pred_m + Kg @ (acc - C @ state_pred_m)
    state_S = (np.eye(9) - Kg @ C) @ state_pred_S + reg*np.eye(9)

    return multivariate_normal(state_m,state_S)

def backshift(B,v):
    B[:,:-1] = B[:,1:]
    B[:,0] = v
    return B

if __name__ == '__main__':

    # Time
    len_trial = 20
    len_horizon = 3;
    now = dtime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # Dimensionalities
    Mu = 2 
    My = 2
    Dy = 6 
    Du = 8
    Dx = My*Dy + (Mu+1)*Du

    # Prior parameters
    Nu0     = 20.
    Omega0  = 1e0*np.diag(np.ones(Dy))
    Lambda0 = 1e-3*np.diag(np.ones(Dx))
    Mean0   = 1e-8*rnd.randn(Dx,Dy)
    Upsilon = 1e-4*np.diag(np.ones(Du))

    # Setpoint (desired observation)
    m_star = [0.0, -10., 0.0, 0.0, 0.0, 0.0] # [yaw, pitch, roll, p_x, p_y, p_z]
    v_star = [1e0, 1e-5, 1e0, 1e3, 1e3, 1e3]
    goal   = multivariate_normal(m_star, np.diag(v_star))

    # Control limits
    u_lims = (-20,20)

    # Preallocate
    times       = np.zeros((len_trial))
    y_          = np.zeros((Dy,len_trial))
    u_          = np.zeros((Du,len_trial))
    Means       = np.zeros((Dx,Dy,len_trial))
    Lambdas     = np.zeros((Dx,Dx,len_trial))
    Omegas      = np.zeros((Dy,Dy,len_trial))
    Nus         = np.zeros((len_trial))

    Means[:,:,0]   = Mean0
    Lambdas[:,:,0] = Lambda0
    Omegas[:,:,0]  = Omega0
    Nus[0]         = Nu0

    yb = np.zeros((Dy,My))
    ub = np.zeros((Du,Mu+1))

    y_ = rnd.randn(Dy,len_trial)
    u_ = np.round(rnd.randn(Du,len_trial))

    IMUstate = np.zeros((9))

    goodPorts = {}
    try:

        # connectPort(goodPorts)    
        # send(goodPorts, ['B', 0.0])
        # send(goodPorts, ['I', [8, 0, 12, 0, 9, 0, 13, 0, 11, 0, 15, 0, 10, 0, 14, 0], 0.0])

        # Start the stopwatch
        times[0] = timeit.default_timer()
        for k in range(1,len_trial):
            times[k] = timeit.default_timer()
            logger.info(f"tstep = {k}/{len_trial}")
            dt = times[k] - times[k-1]

            "Interact with environment"

            # Update system with selected control
            actions = [ 8, u_[0,k-1].astype(int), 
                       12, u_[1,k-1].astype(int),
                        9, u_[2,k-1].astype(int),
                       13, u_[3,k-1].astype(int),
                       11, u_[4,k-1].astype(int),
                       15, u_[5,k-1].astype(int),
                       10, u_[6,k-1].astype(int),
                       14, u_[7,k-1].astype(int)]
            # send(goodPorts, ['I', actions, 0.0])
        
            # Read IMU
            # ypracc = send(goodPorts, ['v', 0])
            # ypracc = np.array(ypracc[1].split()[-6:],dtype='float64')
            # IMUstate = acc2pos(ypracc[-3:]/1e3, IMUstate, dt=dt, reg=np.maximum(1e-3,10*dt))
            # y_[:,k] = np.concatenate([ypracc[:3],IMUstate[:3]])

            # Update ybuffer
            
                    
            "Update parameters"

            call = f"""
                infer_params({y_[:,k].tolist()},
                             {yb[:,0].tolist()},
                             {yb[:,1].tolist()},
                             {ub[:,0].tolist()},
                             {ub[:,1].tolist()},
                             {ub[:,2].tolist()},
                             {Means[:,:,k-1].tolist()},
                             {Lambdas[:,:,k-1].tolist()},
                             {Omegas[:,:,k-1].tolist()},
                             {Nus[k-1].tolist()})
            """
            params = jl.eval(call)

            "Plan actions"

            ##

            # # Impose safety constraints
            # u_[:,k] = np.clip(policy, a_min=u_lims[0], a_max=u_lims[1]).astype(int)
            # logger.info(u_[:,k])

            # Update buffer
            yb = backshift(yb,y_[:,k])
            ub = backshift(ub,u_[:,k])
            

        # closeAllSerial(goodPorts)
        logger.info("Ports closed.")

        print("Saving results..")
        # np.save("results/trials/agent-ARxI-" + now + "-times.npy", times)
        print("Experiment completed.")

    except Exception as e:
        logger.info("Exception")
        logger.info(e)
        # closeAllSerial(goodPorts)
        os._exit(0)

