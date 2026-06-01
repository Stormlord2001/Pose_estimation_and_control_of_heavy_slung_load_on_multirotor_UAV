from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import numpy as np
import casadi as cs
import scipy.linalg

class BodyrateMPC():
    def __init__(self, model):
        # Get model and set up MPC time parameters
        self.model = model
        self.Tf = 1
        self.N = 50

        # Initial state
        #self.x0 = np.array([0.01, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        # Set up the model and solver
        self.ocp_solver, self.integrator = self.setup(self.x0, self.N , self.Tf)
        
    def setup(self, x0, N_horizon, Tf):
        # Create OCP object
        ocp = AcadosOcp()

        # Set the model
        model = self.model.get_acados_model()
        Fmax = self.model.max_thrust
        wmax = self.model.max_rate

        ocp.model = model

        nx = model.x.size()[0]
        nu = model.u.size()[0]
        ny = nx + nu
        ny_e = nx

        # Set dimensions
        ocp.dims.N = N_horizon

        # Create cost matrices
        Q_mat = 2*np.diag([1e1, 1e1, 1e1, 1e1, 1e1, 1e1, 0.0, 0.1, 0.1, 0.1])
        Q_e = 2*np.diag([3e2, 3e2, 3e2, 1e2, 1e2, 1e2, 0.0, 0.0, 0.0, 0.0])
        R_mat = 2*np.diag([1e2, 5e2, 5e2, 5e2])

        # TODO: Add terminal costs

        # the 'EXTERNAL' cost type can be used to define general cost terms
        # NOTE: This leads to additional (exact) hessian contributions when using GAUSS_NEWTON hessian.
        # ocp.cost.cost_type = 'EXTERNAL'
        # ocp.cost.cost_type_e = 'EXTERNAL'
        # ocp.model.cost_expr_ext_cost = model.x.T @ Q_mat @ model.x + model.u.T @ R_mat @ model.u
        # ocp.model.cost_expr_ext_cost_e = model.x.T @ Q_e @ model.x

        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
        ocp.cost.W_e = scipy.linalg.block_diag(Q_e)

        ocp.model.cost_y_expr = cs.vertcat(model.x, model.u)
        ocp.model.cost_y_expr_e = model.x
        ocp.cost.yref  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ocp.cost.yref_e = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        # set constraints
        ocp.constraints.lbu = np.array([0.35*(+Fmax), -wmax, -wmax, -0.5*wmax])
        ocp.constraints.ubu = np.array([+Fmax,  wmax, wmax, 0.5*wmax])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        # Initial conditions
        ocp.constraints.x0 = x0

        # Setup the solver
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        #ocp.solver_options.qp_solveR = 'PARTIAL_CONDENSING_OSQP'

        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'EXACT' or 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'

        # Real-time iteration
        use_RTI = True
        if use_RTI:
            ocp.solver_options.nlp_solver_type = 'SQP_RTI'
            ocp.solver_options.sim_method_num_stages = 4
            ocp.solver_options.sim_method_num_steps = 3
        else:
            ocp.solver_options.nlp_solver_type = 'SQP'

        # Prediction horizon
        ocp.solver_options.tf = Tf

        # Create the solver and integrator object (for simulation)
        ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
        integrator = AcadosSimSolver(ocp, json_file='acados_sim.json')

        return ocp_solver, integrator
    
    def solve(self, x0, verbose=False):
        ocp_solver = self.ocp_solver

        # Set the initial state for the OCP
        ocp_solver.set(0, 'lbx', x0)
        ocp_solver.set(0, 'ubx', x0)

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
