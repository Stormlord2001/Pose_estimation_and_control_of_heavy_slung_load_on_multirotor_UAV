%% Clear workspace
clear all;
clc;

%% Get deafult simulink_opts
simulink_opts = get_acados_simulink_opts;

% Input ports
simulink_opts.inputs.cost_W_0 = 0;
simulink_opts.inputs.cost_W = 0;
simulink_opts.inputs.cost_W_e = 0;
simulink_opts.inputs.x_init = 1;
simulink_opts.inputs.p_global = 0;
simulink_opts.inputs.p = 0;
simulink_opts.inputs.reset_solver = 1;

% Output ports
simulink_opts.outputs.u0 = 1;
simulink_opts.outputs.utraj = 0;
simulink_opts.outputs.xtraj = 1;
simulink_opts.outputs.cost_value = 0;   
simulink_opts.outputs.KKT_residual = 0;
simulink_opts.outputs.KKT_residuals = 0;
simulink_opts.outputs.CPU_time = 1;

simulink_opts.samplingtime = '-1';

%% Run slung_load_ocp
slung_load_ocp;

%% Compule Sfunctions
cd c_generated_code
make_sfun; % ocp solver
cd ..
