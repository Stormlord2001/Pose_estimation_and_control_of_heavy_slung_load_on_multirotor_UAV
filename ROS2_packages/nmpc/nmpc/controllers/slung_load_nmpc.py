from datetime import time
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import numpy as np
import casadi as cs
import scipy.linalg

class SlungLoadNMPC():
    def __init__(self, model):
        # Get model and set up MPC time parameters
        self.model = model
        self.h = 0.02
        self.N = 30

        # Shooting nodes
        # Matlab code: shooting_nodes = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05*(1:N-4)];
        self.shooting_nodes = np.array([i*self.h for i in range(5)] + [0.05 + 0.05*(i-4) for i in range(5, self.N)])
        self.Tf = self.shooting_nodes[-1]

        # Initial state
        # x = {xL, vL, q, w, qe}
        self.x0 = np.zeros(16)
        self.x0[8] = -1.0 # initial z-vector component of cable direction
        self.x0[12] = 1.0 # initial drone quaternion scalar part

        # Set up the model and solver
        self.ocp_solver, self.integrator = self.setup(self.x0, self.N , self.Tf, self.shooting_nodes)
        
    def setup(self, x0, N_horizon, Tf, shooting_nodes):
        # Create OCP object
        ocp = AcadosOcp()

        # Set the model
        model = self.model.get_acados_model()

        # Add model to ocp
        ocp.model = model

        nx = model.x.size()[0]
        nu = model.u.size()[0]
        
        ny_0 = nu
        ny = nx + nu
        ny_e = nx

        # Set dimensions
        ocp.dims.N = N_horizon

        # Create cost matrices
        # x = {xL, vL, q, w, qe}
        # u = {F, w}
        #W_x  = np.diag([1, 1, 1,   1, 1, 1,   1000, 1000, 1000,   1, 1, 1,   1, 1, 1, 1])
        W_x  = np.diag([1e1, 1e1, 1e2,   1e1, 1e1, 1e1,   1e3, 1e3, 1e3,   1e1, 1e1, 1e1,   1e2, 1e1, 1e1, 1e1])
        W_xe = np.diag([3e1, 3e1, 1e3,   1e1, 1e1, 1e1,   1e2, 1e2, 1e2,   1, 1, 1,   2e2, 2e1, 2e1, 2e1])
        #W_u = np.diag([0.1, 0.1, 0.1, 0.1])
        W_u = np.diag([1e1, 5e1, 5e1, 1e1]) 

        # Initial cost term
        ocp.cost.cost_type_0 = 'NONLINEAR_LS'
        ocp.cost.W_0 = W_u
        ocp.cost.yref_0 = np.zeros(ny_0)
        ocp.model.cost_y_expr_0 = model.u
        
        # Path cost term
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.W = scipy.linalg.block_diag(W_x, W_u)
        ocp.cost.yref  = np.zeros(ny)
        ocp.model.cost_y_expr = cs.vertcat(model.x, model.u)

        # Terminal cost term
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        ocp.cost.W_e = W_xe
        ocp.cost.yref_e = np.zeros(ny_e)
        ocp.model.cost_y_expr_e = model.x

        # Set constraints on u
        F_max = self.model.max_thrust
        w_max = self.model.max_rate

        U_min = np.array([0.65*F_max, -w_max, -w_max, -w_max])
        U_max = np.array([0.90*F_max,  w_max, w_max, w_max])

        ocp.constraints.idxbu = np.array([0, 1, 2, 3])
        ocp.constraints.lbu = U_min
        ocp.constraints.ubu = U_max

        # Set constraints on x
        # x = {xL, vL, q, w, qe}
        optitrack_xy_bounds = 5.0
        optitrack_z_bounds = 1.5
        max_vel_xy = 3.0
        max_vel_z = 0.5
        
        # Hard constraints on position and velocity of the load
        ocp.constraints.x0 = x0 # Initial conditions
        ocp.constraints.idxbx = np.array([0, 1, 2, 3, 4, 5]) # position and velocity of load
        ocp.constraints.lbx = np.array([-optitrack_xy_bounds, -optitrack_xy_bounds, 0.1, -max_vel_xy, -max_vel_xy, -max_vel_z])
        ocp.constraints.ubx = np.array([optitrack_xy_bounds,  optitrack_xy_bounds,  optitrack_z_bounds,  max_vel_xy,  max_vel_xy,  max_vel_z])

        # Soften constraints on velocity of the load
        soft_state_indices = np.array([3, 4, 5])  # velocity of load
        ocp.constraints.idxsbx = soft_state_indices
        
        nsbx = len(soft_state_indices)
        ocp.cost.zl =  np.ones(nsbx) * 5e1   # linear penalty
        ocp.cost.zu =  np.ones(nsbx) * 5e1
        ocp.cost.Zl =  np.ones(nsbx) * 5e3   # quadratic penalty
        ocp.cost.Zu =  np.ones(nsbx) * 5e3

        # Nonlinear constraints for swing angle of the cable
        theta_load_max = np.deg2rad(50)  # maximum swing angle in radians
        q = model.x[6:9]  # cable direction vector
        swing_expr = q[0]**2 +q[1]**2 # Should  be less than sin(theta_load_max)^2
        q_norm_error = cs.dot(q, q) - 1.0  # Should be equal to zero

        # Construct constraint
        #ocp.model.con_h_expr_0 = cs.vertcat(swing_expr)
        #ocp.model.con_h_expr = cs.vertcat(swing_expr)
        #ocp.constraints.lh_0 = np.array([0.0])
        #ocp.constraints.uh_0 = np.array([np.sin(theta_load_max)**2])
        #ocp.constraints.lh = np.array([0.0])
        #ocp.constraints.uh = np.array([np.sin(theta_load_max)**2])

        #ocp.model.con_h_expr = cs.vertcat(q_norm_error)
        #ocp.constraints.lh = np.array([0.0])
        #ocp.constraints.uh = np.array([0.0])

        # Soften constraints
        #ocp.constraints.idxsh = np.array([0])  # Index of the nonlinear constraint
        #ocp.constraints.zl = np.array([1e2])  # Lower soft bound
        #ocp.constraints.zu = np.array([1e2])  # Upper soft bound
        #ocp.constraints.Zl = np.array([1e4])  # Penalty weight for lower bound
        #ocp.constraints.Zu = np.array([1e4])  # Penalty weight for upper bound
        


        # Constraint types
        #ocp.constraints.constr_type = 'AUTO'
        #ocp.constraints.constr_type_0 = 'AUTO'  

        ##############################################
        # Setup the solver
        ##############################################
        ocp.solver_options.tf = Tf                                        # Prediction horizon
        #ocp.solver_options.N_horizon = N_horizon                          # Number of intervals
        ocp.solver_options.N_horizon = N_horizon
        #ocp.solver_options.shooting_nodes = shooting_nodes                # Shooting nodes (Non-uniform interval)   
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'                    # Real-time iteration
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'         # QP solver
        ocp.solver_options.qp_solver_cond_N = 5                           # Condensing horizon
        ocp.solver_options.globalization = 'MERIT_BACKTRACKING'           # Globalization strategy
        ocp.solver_options.nlp_solver_max_iter = 1                        # Max NLP solver iterations (1 for RTI)
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'                # Hessian approximation
        ocp.solver_options.integrator_type = 'ERK'                        # Explicit Runge-Kutta integrator 
        ocp.solver_options.sim_method_num_stages = 4                      # Number of stages of the integrator
        ocp.solver_options.sim_method_num_steps = 1                       # Number of steps of the integrator

        # Create the solver and integrator object (for simulation)
        ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
        integrator = AcadosSimSolver(ocp, json_file='acados_sim.json')

        return ocp_solver, integrator
    
    def solve(self, x0, verbose=False):
        ocp_solver = self.ocp_solver

        # Set the initial state for the OCP
        ocp_solver.set(0, 'lbx', x0)
        ocp_solver.set(0, 'ubx', x0)

        for stage in range(self.N-1):
            x_next = ocp_solver.get(stage+1, "x")
            u_next = ocp_solver.get(stage+1, "u")

            ocp_solver.set(stage, "x", x_next)
            ocp_solver.set(stage, "u", u_next)

        ocp_solver.set(self.N, "x", ocp_solver.get(self.N, "x"))

        # Solve the OCP
        status = ocp_solver.solve()

        if verbose:
            self.ocp_solver.print_statistics()

        if status != 0:
            raise Exception(f"AcadosOcpSolver returned status {status}.")

        # Setup lists to hold results
        N = self.N
        nx = self.model.get_acados_model().x.size()[0]
        nu = self.model.get_acados_model().u.size()[0]

        simX = np.ndarray((N+1, nx))
        simU = np.ndarray((N, nu))

        # Get the solution
        for i in range(N):
            simX[i, :] = ocp_solver.get(i, 'x')
            simU[i, :] = ocp_solver.get(i, 'u')
        simX[N, :] = ocp_solver.get(N, 'x')

        return simU, simX
    
    def set_setpoint(self, setpoint):
        # Update the reference trajectory in the OCP solver
        ocp_solver = self.ocp_solver

        N = self.N
        # Inertial reference on u
        y_ref = np.zeros(4)  # ny_0 = nu = 4
        y_ref[0:4] = np.array([self.model.hover_thrust, 0.0, 0.0, 0.0])  # hover thrust and zero angular rates
        ocp_solver.set(0, 'yref', y_ref)

        # Set the reference for each stage
        for i in range(N-1):
            y_ref = np.zeros(20)  # ny = nx + nu = 16 + 4 = 20
            y_ref[0:3] = setpoint  # desired load position
            y_ref[6:9] = np.array([0, 0, -1])   # desired cable direction (hanging down)
            y_ref[12:16] = np.array([1, 0, 0, 0])   # desired drone attitude (no rotation)
            y_ref[16:20] = np.array([self.model.hover_thrust, 0.0, 0.0, 0.0])  # hover thrust and zero angular rates
            ocp_solver.set(i+1, 'yref', y_ref)
        
        # Set the terminal reference
        y_ref_e = np.zeros(16)  # ny_e = nx = 16
        y_ref_e[0:3] = setpoint  # desired load position
        y_ref_e[6:9] = np.array([0, 0, -1])   # desired cable direction (hanging down)
        y_ref_e[12:16] = np.array([1, 0, 0, 0])   # desired drone attitude (no rotation)
        ocp_solver.set(N, 'yref', y_ref_e)

    def circular_reference_trajectory(self, radius, angular_velocity, t_now):
        # Update the reference trajectory in the OCP solver
        ocp_solver = self.ocp_solver
        N = self.N
        
        # Inertial reference on u
        y_ref = np.zeros(4)  # ny_0 = nu = 4
        y_ref[0:4] = np.array([self.model.hover_thrust, 0.0, 0.0, 0.0])  # hover thrust and zero angular rates
        ocp_solver.set(0, 'yref', y_ref)

        for i in range(N-1):
            t_pred = t_now + self.shooting_nodes[i]

            # Circular trajectory
            cx, cy, cz = 0.0, 0.0, 1.0
            R = radius
            w = angular_velocity

            pos_ref = np.array([
                cx + R * np.cos(w * t_pred),
                cy + R * np.sin(w * t_pred),
                cz
            ])

            vel_ref = np.array([
                -R * w * np.sin(w * t_pred),
                R * w * np.cos(w * t_pred),
                0.0
            ])

            y_ref = np.zeros(20)

            # Load position
            y_ref[0:3] = pos_ref

            # Load velocity
            y_ref[3:6] = vel_ref

            # Desired cable direction (down)
            y_ref[6:9] = np.array([0, 0, -1])

            # Desired drone attitude (level and facing forward)
            y_ref[12:16] = np.array([1, 0, 0, 0])

            # Hover thrust & zero body rates
            y_ref[16:20] = np.array([self.model.hover_thrust, 0.0, 0.0, 0.0])

            ocp_solver.set(i+1, 'yref', y_ref)
    
        # Terminal reference
        t_terminal = t_now + self.shooting_nodes[-1]
        pos_ref_terminal = np.array([
            cx + R * np.cos(w * t_terminal),
            cy + R * np.sin(w * t_terminal),
            cz
        ])

        y_ref_e = np.zeros(16)
        y_ref_e[0:3] = pos_ref_terminal
        y_ref_e[6:9] = np.array([0, 0, -1])
        y_ref_e[12:16] = np.array([1, 0, 0, 0])

        ocp_solver.set(N, 'yref', y_ref_e)
