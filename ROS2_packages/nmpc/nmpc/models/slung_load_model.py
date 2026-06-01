from acados_template import AcadosModel
import casadi as cs

class SlungLoadModel():
    def __init__(self):
        self.name = 'slung_load_model'

        # Constants
        self.L = 3.0
        self.mQ = 1.8306
        self.mL = 0.4623
        self.mass = self.mQ + self.mL

        self.hover_throttle = 0.71
        self.hover_thrust = (self.mass) * 9.82
        self.max_thrust = self.hover_thrust / self.hover_throttle
        self.max_rate = 0.25

        self.g = 9.82
    
    def get_acados_model(self) -> AcadosModel:
        # Define states and control input variables of the model
        xL = cs.SX.sym('xL', 3)      # load position
        vL = cs.SX.sym('vL', 3)      # load velocity
        q = cs.SX.sym('q',3)         # cable direction (unit vector from drone to load)
        wL = cs.SX.sym('wL',3)       # angular velocity of the load in inertial frame
        qe = cs.SX.sym('qe',4)       # drone attitude quaternion
        x = cs.vertcat(xL, vL, q, wL, qe)

        F = cs.SX.sym('F')           # total thrust of the quadcopter
        w = cs.SX.sym('w',3)         # desired angular velocity of the quadcopter
        u = cs.vertcat(F, w)

        # Define symbolic variables for state derivatives
        xdot = cs.SX.sym('xdot', 16)

        # Quaternion to rotation matrix
        def q_to_rot_mat(q):
            qs = q[0]
            qv = q[1:4]

            I3 = cs.SX.eye(3)

            hat_qv = skew_symmetric(qv)

            R  = (qs**2 - cs.mtimes(qv.T, qv)) * I3 + 2 * cs.mtimes(qv, qv.T) + 2 * qs * hat_qv

            return R
        
        def skew_symmetric(v):
            return cs.vertcat(
                cs.horzcat(0, -v[2], v[1]),
                cs.horzcat(v[2], 0, -v[0]),
                cs.horzcat(-v[1], v[0], 0))
        
        def big_skew_symmetric(v):
            return cs.vertcat(
                cs.horzcat(0, -v[0], -v[1], -v[2]),
                cs.horzcat(v[0], 0, v[2], -v[1]),
                cs.horzcat(v[1], -v[2], 0, v[0]),
                cs.horzcat(v[2], v[1], -v[0], 0)
            )
        
        def quat_multiply(q, p):
            return cs.vertcat(
                q[0]*p[0] - q[1]*p[1] - q[2]*p[2] - q[3]*p[3],
                q[0]*p[1] + q[1]*p[0] + q[2]*p[3] - q[3]*p[2],
                q[0]*p[2] - q[1]*p[3] + q[2]*p[0] + q[3]*p[1],
                q[0]*p[3] + q[1]*p[2] - q[2]*p[1] + q[3]*p[0]
            )
        
        # Rotation matrix from body to inertial frame
        R = q_to_rot_mat(qe)

        # Dynamics equations
        e3 = cs.vertcat(0, 0, 1)

        xL_dot = vL

        q_dot = cs.mtimes(skew_symmetric(wL), q)
        
        #q_dot_sq = cs.dot(q, q)
        q_dot_sq = cs.dot(q_dot, q_dot)
        thrust_vec = F * (R @ e3)
        lhs_mass = self.mass
        scalar = cs.dot(q, thrust_vec) - self.mQ*self.L*q_dot_sq
        vL_dot = cs.mtimes(scalar, q)/lhs_mass - self.g*e3

        wL_dot = (-cs.cross(q, thrust_vec))/(self.mQ*self.L)

        #qe_dot = 0.5 * cs.mtimes(big_skew_symmetric(w), qe)
        qe_dot = 0.5 * quat_multiply(qe, cs.vertcat(0, w))

        f_expl = cs.vertcat(
            xL_dot, #3
            vL_dot, #3
            q_dot,  #3
            wL_dot, #3
            qe_dot  #4
        )

        f_impl = xdot - f_expl

        model = AcadosModel()
        model.f_expl_expr = f_expl
        model.f_impl_expr = f_impl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.name = self.name

        return model
