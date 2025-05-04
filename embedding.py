import numpy as np
from scipy.integrate import solve_ivp
from manim import *

class SimpleEmbedding(ThreeDScene):
    def func(self, u, v):
       return np.array([u, v, u**2])

    def construct(self):
        # add 3d axes
        axes = ThreeDAxes()
        self.add(axes)

        mu = -0.05
        l = -1

        def system(t, y):
            return [mu*y[0], l * (y[1] - y[0]**2)]

        # create 40 random points in -2 2 -2 2

        self.set_camera_orientation(0.50, .75)

        #y_0s = np.array([np.linspace(-2, 2, 10), np.linspace(-2, 2, 10)]).T
        x, y = np.linspace(-2, 2, 20), np.linspace(-2, 2, 5)
        #print(y_0s.size())
        trajs = []
        for x_0 in x:
            for y_0 in y:
                sol = solve_ivp(system, [0, 20], [x_0, y_0], t_eval=np.linspace(0, 10, 100), rtol=1e-10).y
                traj = axes.plot_line_graph(sol[0], sol[1], sol[0]**2,
                                            line_color=BLUE_A,
                                            stroke_width=2,
                                            add_vertex_dots=False,
                )
                trajs.append(traj)


        surface = Surface(
            lambda u, v: axes.c2p(*self.func(u, v)),
            u_range=[-PI, PI],
            v_range=[0, TAU],
            resolution=8,
        )
        self.set_camera_orientation(theta=70 * DEGREES, phi=75 * DEGREES)
        self.add(surface)


        # animate traj creation
        #self.play(Create(traj, rate_functions=linear), run_time=5)
        self.play([Create(traj, rate_functions=linear) for traj in trajs], run_time=15)
