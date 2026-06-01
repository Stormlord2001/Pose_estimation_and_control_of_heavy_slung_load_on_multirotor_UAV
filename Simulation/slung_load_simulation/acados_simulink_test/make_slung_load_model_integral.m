function model = make_slung_load_model()
    import casadi.*
    
    % x = {xL, vL, q, w, qe, Om, z_int}

    % --- Symbolic variables ---
    x = SX.sym('x',20,1);
    u = SX.sym('u',4,1);
    p = SX.sym('p',8,1);  % mQ,mL,l,g,Jx,Jy,Jz, z_ref
    xdot = SX.sym('xdot',20,1);
    
    % --- Unpack states, inputs, params (same as before) ---
    xL = x(1:3); vL = x(4:6); q = x(7:9); w = x(10:12);
    qe = x(13:16); Om = x(17:19);
    z_int = x(20);
    f_th = u(1); M = u(2:4);
    mQ=p(1); mL=p(2); l=p(3); g=p(4); Jx=p(5); Jy=p(6); Jz=p(7); z_ref=p(8);
    J = diag([Jx,Jy,Jz]);
    e3=[0;0;1];
    
    % --- Quaternion to rotation matrix (scalar-first) ---
    qs = qe(1); qv = qe(2:4);
    I3 = SX.eye(3);
    hat_qv = [0 -qv(3) qv(2); qv(3) 0 -qv(1); -qv(2) qv(1) 0];
    R = (qs^2 - (qv'*qv))*I3 + 2*(qv*qv') + 2*qs*hat_qv;
    
    % --- Dynamics ---
    xdot_xL = vL;
    xdot_q  = cross(w,q);
    vQ = vL - l*xdot_q;
    qdot_sq = dot(xdot_q,xdot_q);
    thrust_vec = f_th * (R*e3);
    lhs_mass = mQ+mL;
    scalar = dot(q,thrust_vec) - mQ*l*qdot_sq;
    xdot_vL = (scalar*q)/lhs_mass - g*e3;
    xdot_w = (-cross(q,thrust_vec))/(mQ*l);
    
    Omega_quat = SX.zeros(4,4);
    Omega_quat(1,2:4) = -Om';
    Omega_quat(2:4,1) = Om;
    Omega_quat(2:4,2:4) = -[0 -Om(3) Om(2); Om(3) 0 -Om(1); -Om(2) Om(1) 0];
    xdot_qe = 0.5 * Omega_quat * qe;
    
    xdot_Om = J \ (M - cross(Om,J*Om));
    
    % --- Integral state dynamics ---
    xdot_zint = z_ref - xL(3);
    
    f_expl = vertcat(xdot_xL,xdot_vL,xdot_q,xdot_w,xdot_qe,xdot_Om, xdot_zint);
    
    % --- Create acados_model object ---
    model = AcadosModel();
    model.name = 'slung_load_model_integral';
    model.x = x;
    model.xdot = xdot;
    model.u = u;
    model.p = p;
    model.f_expl_expr = f_expl;
    model.f_impl_expr = xdot - f_expl;
end
