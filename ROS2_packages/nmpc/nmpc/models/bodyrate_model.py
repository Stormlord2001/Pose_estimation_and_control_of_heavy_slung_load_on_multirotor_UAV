from acados_template import AcadosModel
import casadi as cs

# Model for a quadrotor using bodyrtate as input and using quaternions to describe attitude
class BodyrateModel():
    def __init__(self):
        self.name = 'bodyrate_model'

        # Constants
        #self.mass = 1
        self.mass = 1.777

        #hover_thrust = 0.73
        hover_thrust = 0.5
        #hover_thrust = 0.3
        self.max_thrust = self.mass * 9.82/hover_thrust
        self.max_rate = 0.1

    def get_acados_model(self) -> AcadosModel:
        def skew_symmetric(v):
            return cs.vertcat(
                cs.horzcat(0, -v[0], -v[1], -v[2]),
                cs.horzcat(v[0], 0, v[2], -v[1]),
                cs.horzcat(v[1], -v[2], 0, v[0]),
                cs.horzcat(v[2], v[1], -v[0], 0))
        
        def q_to_rot_mat(q):
            qw, qx, qy, qz = q[0], q[1], q[2], q[3]

            rot_mat = cs.vertcat(
                cs.horzcat(1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
                cs.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)),
                cs.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)))

            return rot_mat

        def v_dot_q(v, q):
            rot_mat = q_to_rot_mat(q)

            return cs.mtimes(rot_mat, v)
        
        model = AcadosModel()

        # Defines the states and control input variables of the model
        # states - p: position, v: velocity, q: quaternion attitude
        p = cs.MX.sym('p', 3)
        v = cs.MX.sym('v', 3)
        q = cs.MX.sym('q', 4)
        x = cs.vertcat(p, v, q)

        # control input - F: thrust, w: angular velocity
        F = cs.MX.sym('F')
        w = cs.MX.sym('w', 3)
        u = cs.vertcat(F, w)

        # State derivatives
        p_dot = cs.MX.sym('p_dot', 3)
        v_dot = cs.MX.sym('v_dot', 3)
        q_dot = cs.MX.sym('q_dot', 4)
        x_dot = cs.vertcat(p_dot, v_dot, q_dot)

        # Gravity force in inertial frame
        g = cs.vertcat(0, 0, -9.82)

        # Thrust force in body frame
        force = cs.vertcat(0, 0, F)

        # Acceleration in inertial frame due to thrust
        a_thrust = v_dot_q(force, q) / self.mass

        # Dynamic equations
        f_expl = cs.vertcat(
            v,
            a_thrust + g,
            1/2 * cs.mtimes(skew_symmetric(w), q)
        )

        f_impl = x_dot - f_expl

        model.f_expl_expr = f_expl
        model.f_impl_expr = f_impl
        model.x = x
        model.xdot = x_dot
        model.u = u
        model.name = self.name

        return model
