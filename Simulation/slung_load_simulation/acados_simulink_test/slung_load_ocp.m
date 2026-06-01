import casadi.*
% Needed for simulink
if ~exist('simulink_opts','var')
    disp('using empty simulink_opts to generate solver without simulink block')
    simulink_opts = [];
end

check_acados_requirements()

%% Model parameters
mQ = 1.8306; mL = 0.4623; g = 9.82; l = 3.0;

%% Solver settings
% initial state
x0 = zeros(20,1);
x0(3) = 0;                 % load z-position
x0(7:9) = [0; 0; -1];       % Cable direction q

% Initial rotation of load in inertial frame
yaw = 0;
pitch = 0;
roll = 0;
q = angle2quat(yaw, pitch, roll);
x0(13:16) = q;

% Shooting nodes
h = 0.02; % sampling time = length of first shooting interval
N = 30; % number of shooting intervals
% nonuniform discretization
shooting_nodes = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05*(1:N-4)];
T = shooting_nodes(end);

%% Model dynamics
model = make_slung_load_model_integral();

% dimensions
nx = model.x.rows();
nu = model.u.rows();

%% OCP formulation object
ocp = AcadosOcp();
ocp.model = model;

%% COST: nonlinear-least squares cost
% x = {xL, vL, q, w, qe, Om, z_int}
W_int = 0; % Integral weight: 0 -> disabled
W_x = diag([10*ones(1,2), 100*ones(1,1), 0.1*ones(1,3), 1000*ones(1,3), 1*ones(1,3), 100*ones(1,1), 10*ones(1,3), 1*ones(1,3), W_int]);
W_xe = diag([30*ones(1,2), 1000*ones(1,1), 0.1*ones(1,3), 100*ones(1,3), 1*ones(1,3), 200*ones(1,1), 20*ones(1,3), 1*ones(1,3), W_int]);
W_u = diag([10,50,50,10]);

% Initial cost term
ny_0 = nu;
ocp.cost.cost_type_0 = 'NONLINEAR_LS';
ocp.cost.W_0 = W_u;
ocp.cost.yref_0 = zeros(ny_0, 1);
model.cost_y_expr_0 = model.u;

% Path cost term
ny = nx + nu;
ocp.cost.cost_type = 'NONLINEAR_LS';
ocp.cost.W = blkdiag(W_x, W_u);
ocp.cost.yref = zeros(ny, 1);
model.cost_y_expr = vertcat(model.x, model.u);

% Terminal cost term
ny_e = nx;
ocp.cost.cost_type_e = 'NONLINEAR_LS';
ocp.cost.W_e = W_xe;
ocp.cost.yref_e = zeros(ny_e, 1);
model.cost_y_expr_e = model.x;

%% Define constraints
f_hover = (mL + mQ) * g;
f_min = 0.65 * f_hover; 
f_max = 0.9 * f_hover;
M_max = 1;

% Constraints on U
U_max = [f_max, M_max, M_max, M_max];
U_min = [f_min, -M_max, -M_max, -M_max];

ocp.constraints.constr_type = 'AUTO';
ocp.constraints.constr_type_0 = 'AUTO';

model.con_h_expr_0 = model.u;
ocp.constraints.lh_0 = U_min;
ocp.constraints.uh_0 = U_max;

model.con_h_expr = model.u;
ocp.constraints.lh = U_min;
ocp.constraints.uh = U_max;


% Constraints on X
% x = {xL, vL, q, w, qe, Om, z_int}
% No initial state constraint
ocp.constraints.x0 = zeros(nx,1);


% State constraints
ocp.constraints.idxbx = [1 2 3 4 5 6 17 18 19]-1;
ocp.constraints.lbx = [-5, -5, -5, -3, -3, -3, -0.25, -0.25, -0.25];
ocp.constraints.ubx = [5, 5, 5, 3, 3, 3, 0.25, 0.25, 0.25];
%{
% Terminal state constraints
ocp.constraints.idxbx_e = ocp.constraints.idxbx;
ocp.constraints.lbx_e = ocp.constraints.lbx;
ocp.constraints.ubx_e = ocp.constraints.ubx;
%}

ocp.constraints.x0 = x0;

%% Solver settings
ocp.solver_options.N_horizon = N;
ocp.solver_options.tf = T;
%ocp.solver_options.shooting_nodes = shooting_nodes;
ocp.solver_options.nlp_solver_type = 'SQP_RTI';
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM';
ocp.solver_options.qp_solver_cond_N = 5; % for partial condensing
ocp.solver_options.globalization = 'MERIT_BACKTRACKING'; % turns on globalization 'MERIT_BACKTRACKING'
ocp.solver_options.nlp_solver_max_iter = 1; %'1000' is one because of RTI
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON';
ocp.solver_options.ext_fun_compile_flags = '-O2';
ocp.simulink_opts = simulink_opts;

% integrator model
ocp.solver_options.integrator_type = 'ERK';
ocp.solver_options.sim_method_num_stages = 4;
ocp.solver_options.sim_method_num_steps = 3;

% Create solver
ocp_solver = AcadosOcpSolver(ocp);
%p = [mQ; mL; l; g; 0.01; 0.01; 0.02; 0];
%ocp_solver.set('p', p);


