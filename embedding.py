import numpy as np
from scipy.integrate import solve_ivp
from manim import *

#class StreamLinesFuncZ(StreamLines):
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)

BLUE_Z_1 = "#669bbc"
BLUE_Z_2 = "#003049"
RED_Z_1 = "#c1121f"
RED_Z_2 = "#780000"
PURPLE_Z_1 = "#d0a5c0"
PURPLE_Z_2 = "#c64191"

def duffing_system(t, state):
    x, y, z = state
    dxdt = y
    dydt = x - x**3
    return [dxdt, dydt, 0]

# taken from https://github.com/BenEcon/manim--/blob/main/lorenz_attractor.py
# Define the Lorenz system of differential equations
def lorenz_system(t, state, sigma=10, rho=28, beta=8 / 3):
    x, y, z = state  # Unpack the state vector into x, y, z
    dxdt = sigma * (y - x)  # Compute the derivative of x
    dydt = x * (rho - z) - y  # Compute the derivative of y
    dzdt = x * y - beta * z  # Compute the derivative of z
    return [dxdt, dydt, dzdt]  # Return the derivatives as a list

# Function to compute solution points for an ODE
def ode_solution_points(function, state0, time, dt=0.01):
    solution = solve_ivp(
        function,  # The ODE function to solve
        t_span=(0, time),  # Time span for the solution
        y0=state0,  # Initial state
        t_eval=np.arange(0, time, dt)  # Time points at which to store the solution
    )
    return solution.y.T  # Return the transposed solution array

class DuffingOscillator(ThreeDScene):
    def construct(self):
        # Set up 3D axes with specified ranges and color
        system = lambda y: np.array([y[1], y[0] - y[0]**3, 0])

        axes = ThreeDAxes(
            x_range=[-1.5, 1.5, 0.1],  # X-axis range from -10 to 10 with tick marks every 1 unit
            y_range=[-1.5, 1.5, 0.1],  # Y-axis range from -1 to 1 with tick marks every 0.1 units
            z_range=[0, 1, 0.1],    # Z-axis range from 0 to 1 with tick marks every 0.1 units
            axis_config={"color": GREY},  # Set the color of the axes to grey
        )



        stream_lines = StreamLines(
            system,
            x_range=[-1.0, 1.0, 0.1],
            y_range=[-0.8, 0.8, 0.2],
            stroke_width=2,
            max_anchors_per_line=100,
            padding=1,
            virtual_time=5,  # Set the virtual time for the stream lines
            colors=[BLUE_Z_2, BLUE_Z_1],
        )

        axes.move_to(ORIGIN)
        stream_lines.fit_to_coordinate_system(axes)

        self.add(stream_lines)
        stream_lines.start_animation(warm_up=True, flow_speed=1, time_width=1.0)
        self.wait(stream_lines.virtual_time / stream_lines.flow_speed)
        self.next_section("loop")
        self.wait(stream_lines.virtual_time / stream_lines.flow_speed)

        #self.add(stream_lines, spawning_area, *labels)

# Define a class for visualizing the Lorenz attractor
class LorenzAttractor(ThreeDScene):
    def construct(self):
        # Set up 3D axes with specified ranges and color
        axes = ThreeDAxes(
            x_range=(-50, 50, 5),  # X-axis range from -50 to 50 with tick marks every 5 units
            y_range=(-50, 50, 5),  # Y-axis range from -50 to 50 with tick marks every 5 units
            z_range=(0, 50, 5),    # Z-axis range from 0 to 50 with tick marks every 5 units
            axis_config={"color": GREY},  # Set the color of the axes to blue
        )

        # Move the axes to the origin of the scene
        axes.move_to(ORIGIN)

        # Set the initial camera orientation with specific angles
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)

        # Add the axes to the scene
        self.add(axes)

        # Define the equations
        """equations = MathTex(
            r"\sigma", r"\rho", r"\beta",
            tex_to_color_map={r"\sigma": YELLOW, r"\rho": ORANGE, r"\beta": PURPLE},
            font_size=40
        )"""
        eq1 = MathTex(r"\frac{dx}{dt} = \sigma \left( y - x \right)", font_size=15)
        eq2 = MathTex(r"\frac{dy}{dt} = x \left( \rho - z \right) - y", font_size=15)
        eq3 = MathTex(r"\frac{dz}{dt} = xy - \beta z", font_size=15)

        equations = VGroup(eq1, eq2, eq3)


        equations.arrange(DOWN)
        equations.to_corner(UL)
        equations.set_stroke(width=1)
        # Add the equation as a fixed object in the frame
        self.add_fixed_in_frame_mobjects(equations)

        # Define parameters for the Lorenz system solutions
        epsilon = 1e-5  # Small perturbation for initial conditions
        evolution_time = 30  # Total time for the system evolution
        n_points = 10  # Number of initial states to simulate
        states = [
            [10, 10, 10 + n * epsilon]  # Initial states with slight variations in z
            for n in range(n_points)
        ]
        colors = color_gradient([BLUE_E, BLUE_A], len(states))  # Generate a gradient of colors

        # Create curves for the Lorenz attractor
        curves = VGroup()
        for state, color in zip(states, colors * 10):  # Repeat colors to match the number of states
            points = ode_solution_points(lorenz_system, state, evolution_time)  # Compute solution points
            # Ensure points are in the correct shape
            if points.shape[1] != 3:
                raise ValueError("Points should have three columns for x, y, z coordinates.")
            # Create a smooth curve from the solution points
            curve = VMobject().set_points_smoothly([axes.c2p(x, y, z) for x, y, z in points])
            curve.set_stroke(color, 1, opacity=0.5)  # Set the stroke color and opacity of the curve
            curves.add(curve)  # Add the curve to the group

        # Ensure curves is not empty and contains valid VMobjects
        if curves:
            valid_curves = [curve for curve in curves if isinstance(curve, VMobject)]
            if not valid_curves:
                raise ValueError("No valid VMobjects found in curves.")
            
            # Move the camera before playing animations
            self.move_camera(theta=PI/2, run_time=10)  # Camera movement

            self.play(
                *(
                    Create(curve, rate_func=linear)  # Use a smooth rate function for smoother animation
                    for curve in valid_curves
                ),
                Write(equations),  # Animate the writing of the equation
                run_time=evolution_time/2
            )
        else:
            raise ValueError("Curves is empty or contains invalid elements.")

        # Create dots to move along the trajectories of the curves
        dots = VGroup(
            *[Dot(color=color, radius=0.1).set_opacity(0.8) for color in colors * 10]
        )

        # Define an updater function to move dots along the curves
        def update_dots(dots, curves=curves):
            for dot, curve in zip(dots, curves):
                dot.move_to(curve.get_end())  # Move each dot to the end of its corresponding curve

        dots.add_updater(update_dots)  # Add the updater to the dots

        # Create traced paths for the dots to leave trails
        tails = VGroup(
            *[
                TracedPath(dot.get_center, stroke_color=dot.get_color(), stroke_width=1)
                for dot in dots
            ]
        )

        #self.add(dots)  # Add the dots to the scene
        self.add(tails)  # Add the tails to the scene
        #curves.set_opacity(0)  # Make the curves invisible
        #self.move_camera(theta=PI/4, run_time=10)  # Camera movement


        # Fade out all tails except the last one
        #for tail in tails[:-1]:
        #    tail.add_updater(lambda m: m.set_opacity(0.5))  # Adjust opacity dynamically

        # Ensure the last tail remains fully visible
        #tails[-1].set_opacity(1)

        self.wait(2)  # Add a wait time to ensure the scene is fully rendered

        # Optionally, you can add a final wait to keep the scene visible
        #self.wait(2)  # Additional wait time to keep the scene visible at the end

class Intro(ThreeDScene):
    def construct(self):
        spacing = 1.25
        welcome = Text("Intro to:", font_size=72).to_edge(UP, buff=spacing)
        koopman = MathTex(r"\mathcal{K}", font_size=2*72).next_to(welcome, DOWN, buff=spacing)
        title = Text("The Koopman Operator", font_size=72).next_to(koopman, DOWN, buff=spacing)

        self.add(welcome)
        self.add(koopman)
        self.add(title)

class Outline(ThreeDScene):
    def construct(self):
        spacing = 0.25
        title = Text("Outline", font_size=72).to_edge(UP, buff=spacing)
        # TODO: The height is different for each item, so we need to set the height of the items to be the same
        items = VGroup(
            Tex("1. Motivation", font_size=72),
            Tex("2. Koopman 101", font_size=72),
            Tex("3. Simple embedding", font_size=72),
            Tex("4. Applications", font_size=72),
            Tex("5. Limitations", font_size=72)
        )
        items.arrange(DOWN, buff=spacing, aligned_edge = LEFT).next_to(title, DOWN, buff=spacing*2).to_edge(LEFT, buff=spacing)
        

        rect = SurroundingRectangle(items, color=WHITE, buff=0.1)

        self.add(title)
        self.add(items)
class Motivation(ThreeDScene):
    def construct(self):
        return

class Derivation(ThreeDScene):
    def construct(self):
        template = TexTemplate()
        template.add_to_preamble(r"\usepackage{mathtools}")
        # Full aligned block
        assumptions = MathTex(
            r"&{\bf x}\in\mathcal{X}, \mathcal{X}\subseteq\mathbb{R}^n, t\in \mathbb{N} \\",
        ).to_corner(UL, buff=0.5)
        state_eqs = MathTex(
            r"{{\frac{d}{dt}{\bf x}(t)}}{{ &= }}{{{\bf f}({\bf x})}} \\"
            r"{{ \textbf{x}_{k+1} }} {{ &= }} {{ {\bf F}({\bf x}_k) }} \\"
            r"{\bf x}(t) &= {\bf F}^t({\bf x}(0)) \\"
            r"{\bf x}(t) &= {\bf F}({\bf F}(\dots({\bf F}({\bf x}(0))))) \\"
        ).to_edge(LEFT, buff=0.5)
        goal_eq = MathTex(
            r"{\bf z}_{k+1} &= {\bf K}{\bf z}_k"
        ).to_corner(DL,buff=1.0)

        definitions = MathTex(
            r"\mathcal{G}(\mathcal{X}),\mathcal{K}^t \colon \mathcal{G}(\mathcal{X}) \to \mathcal{G}(\mathcal{X})"
        ).to_corner(UR, buff=0.5)

        measurement_eqs = MathTex(
            r"g({\bf x}(t)) &= g({\bf F}^t({\bf x}(0))) \\"
            r"\mathcal{K}^tg({\bf x}) &= g({\bf F}^t({\bf x})) \\"
            r"g_t &\coloneqq \mathcal{K}^t g , g_0\coloneqq g \\"
            r"g_t &= \mathcal{K}\circ\mathcal{K}\circ\dots\circ\mathcal{K}g\\"
        , tex_template=template
        ).to_edge(RIGHT, buff=0.5)

        finish_eq = MathTex(
            r"g_{k+1} = \mathcal{K}g_k"
        ).to_corner(DR, buff=1.0)

        self.add(assumptions, state_eqs, goal_eq, measurement_eqs, finish_eq, definitions)

class EigenfunctionsDef(ThreeDScene):
    def construct(self):
        comparision = Tex(r"Similar to eigenvectors for matricies, \\" 
            r"there are \textit{eigenfunctions} $\varphi({\bf x})$ of $\mathcal{K}$, \\"
            r"with corresponding \textit{eigenvalues} $\lambda\in\mathbb{C}$").to_corner(UL, buff=0.5)
        def_eig_fun = MathTex(r"\varphi({\bf x}_{k+1})={\cal K}\varphi({\bf x}_{k})=\lambda\varphi({\bf x}_{k}).")

        expl = Tex(
            r"These functions are time-invariant directions in our observable space $\mathcal{G}(\mathcal{X})$"
            ).to_corner(DR, buff=0.5)

#        self.play(Write(comparision))
#        self.play(Write(def_eig_fun))
#        self.play(Write(expl))
#
#        self.play(FadeOut(comparision))
#        self.play(def_eig_fun.animate.to_edge(UP),
#        expl.animate.to_edge(UP, buff=2))
        self.add(comparision, def_eig_fun, expl)

class Eigenfunctions(ThreeDScene):
    def construct(self):

        multipl = MathTex(
            r"{\cal K}(\varphi_{1}({\bf x})\varphi_{2}({\bf x}))&=\varphi_{1}({\bf F}({\bf x}))\varphi_{2}({\bf F}({\bf x}))\\",
            r"&=\lambda_{1}\lambda_{2}\varphi_{1}({\bf x})\varphi_{2}({\bf x})"
        ).to_edge(UP)

        expl_mult = Tex(
            r"The product of two Eigenfunctions is an Eigenfunctions \\",
            r"(if ${\cal G}({\cal X})$ is closed under multiplication)"
        ).next_to(multipl, DOWN)
        self.add(multipl, expl_mult)

        evolve_lin = MathTex(r"g({\bf x})=\sum_{k}v_{k}\varphi_{k}\quad\Longrightarrow\quad {\cal K}^{t}g({\bf x})=\sum_{k}v_{k}\lambda_{k}^{t}\varphi_{k}")
        expl_evolve = Tex(
            r"Observables $g\in span\{\varphi_k\}^K_{k=1}$ evolve particulary simple \\",
            r"$\Longrightarrow\quad span\{\varphi_k\}^K_{k=1}\subseteq{\cal G}({\cal X})$ is invariant under the action of ${\cal K}$"
            ).next_to(evolve_lin, DOWN)
        self.add(evolve_lin, expl_evolve)

class KMD(ThreeDScene):
    def construct(self):
        measurements = Matrix([[r"g_1({\bf x})"], [r"g_2({\bf x})"], [r"\vdots"],[r"g_p({\bf x})"]], h_buff=0.5, element_alignment_corner=[0, 0, 0])
        single_exp = MathTex(
            r"g_{i}({\bf x})=\sum_{j=1}^{\infty}v_{i j}\varphi_{j}({\bf x})"
        )
        measurements_exp = MathTex(
            r"=\sum_{j=1}^{\infty}\varphi_{j}(\mathbf{x})\mathbf{v}_{j}"
        )
        repr_dyn = MathTex(
            r"{\bf g}({\bf x}(t))={\cal K}^t{\bf g}({\bf x}_0)&={\cal K}^t\sum_{j=1}^{\infty}\varphi_j({\bf x}_0){\bf v}_j\\",
            r"&=\sum_{j=1}^{\infty}{\cal K}^{t}\varphi_{j}({\bf x}_{0}){\bf v}_j\\ ",
            r"&=\sum_{j=1}^{\infty}\lambda_{j}^{t}\varphi_{j}({\bf x}_{0}){\bf v}_j",
        ).to_edge(RIGHT)

        group = VGroup(measurements, measurements_exp).arrange(RIGHT)
        group.to_edge(LEFT, buff=0.5)

        single_exp.next_to(group, UP)




        self.add(group, repr_dyn, single_exp)


class AbstractEmbedding(ThreeDScene):
    def construct(self):
        state_descr = VGroup([Tex("Finite", font_size=72),Tex("nonlinear", font_size=72),Tex("state"," space", font_size=72)])
        state_descr.arrange(DOWN, buff=0.25).to_corner(UL, buff=0.5)
        state_descr[0].set_color(GREEN)
        state_descr[1].set_color(RED)
        state_descr[2][0].set_color(YELLOW_C)
        x = MathTex(r"x \in \mathbb{R}^n", font_size=72).next_to(state_descr, DOWN, buff=0.5)
        arr = Arrow(buff=0.5)
        state_vec = Matrix([["x_1"], ["x_2"], ["\\vdots"],["x_n"]], h_buff=0.5, element_alignment_corner=[0, 0, 0]).next_to(x, DOWN, buff=0.5)
        self.add(state_descr, x, arr, state_vec)

        state_descr2 = state_descr.copy()
        x2 = x.copy()
        state_vec2 = state_vec.copy()

        observ_descr = VGroup([Tex("Ininite", font_size=72),Tex("linear", font_size=72),Tex("observable"," space", font_size=72)])
        observ_descr.arrange(DOWN, buff=0.25).to_corner(UR, buff=0.5)
        observ_descr[0].set_color(RED)
        observ_descr[1].set_color(GREEN)
        observ_descr[2][0].set_color(PURPLE_A)
        phi = MathTex(r"\Phi \in (L^2)^\infty", font_size=72).next_to(observ_descr, DOWN, buff=0.5)
        observ_vec = Matrix([["\phi_1(x)"], ["\phi_2(x)"], ["\\vdots"]], h_buff=0.5, element_alignment_corner=[0, 0, 0]).next_to(phi, DOWN, buff=0.5)
        #self.add(observ_descr, phi, observ_vec)
        self.play(
            AnimationGroup(
                TransformMatchingTex(state_descr2[2], observ_descr[2], run_time=2, fade_transform_mismatches=True),
                TransformMatchingTex(x2, phi, run_time=2, fade_transform_mismatches=True),
                Transform(state_vec2, observ_vec, run_time=2, fade_transform_mismatches=True),
                )
            )
        self.play(TransformMatchingTex(state_descr2[0], observ_descr[0], run_time=2, fade_transform_mismatches=True))
        self.play(TransformMatchingTex(state_descr2[1], observ_descr[1], run_time=2, fade_transform_mismatches=True))
        self.interactive_embed()

class SimpleEmbedding(ThreeDScene):
    def observable(self, u, v):
       return np.array([u, v, u**2])

    def construct(self):
        ### SETUP ###
        axes = ThreeDAxes(z_range=[0, 1, 1], x_range=[-1, 1, 1], y_range=[-1, 1, 1])
        axes.set_color(BLACK)
        axes.set_stroke(width=0.5)
        axes.set_opacity(0.5)
        mu = -0.05
        l = -1

        system = lambda y: np.array([mu*y[0], l * (y[1] - y[0]**2), 0])

        stream_lines = StreamLines(
            system,
            x_range=[-1, 1, 0.05],
            y_range=[-1, 1, 0.05],
            stroke_width=2,
            max_anchors_per_line=100,
            padding=1,
            colors=[BLUE_Z_2, BLUE_Z_1],
        )

        embedding_surface = Surface(
            lambda u, v: axes.c2p(*self.observable(u, v)),
            u_range=[-1, 1],
            v_range=[-1, 1],
            resolution=8,
            fill_opacity=0.2,
            checkerboard_colors=[PURPLE_Z_1, PURPLE_Z_2]
        )

        linear_surface = Surface(
            lambda u, v: axes.c2p(u, v, v),
            u_range=[-1, 1],
            v_range=[0, 1],
            resolution=8,
            fill_opacity=0.6,
            checkerboard_colors=[RED_Z_1, RED_Z_2]

        )

        stream_lines_emb = stream_lines.copy()
        for line in stream_lines_emb:
            for point in line.points:
                point[2] = point[0]**2
        stream_lines.fit_to_coordinate_system(axes)
        stream_lines_emb.fit_to_coordinate_system(axes)

        # time for stream lines to complete one cycle
        cycle_time = stream_lines.virtual_time
        ### ANIMATION ###

        self.add(axes)

        ## Show nonlinear system
        self.add(stream_lines)
        stream_lines.start_animation(warm_up=False, flow_speed=1)
        self.wait(cycle_time)

        ## Move to 3d
        self.move_camera(phi=39 * DEGREES, theta=-45 * DEGREES,frame_center=[0,0,0], run_time=cycle_time)

        ## Introduce embedding

        self.play(FadeIn(embedding_surface), run_time=cycle_time)

        ## Show embedding
        #stream_lines_emb.start_animation(warm_up=False, flow_speed=1)
        stream_lines.end_animation()
        self.play(Transform(stream_lines, stream_lines_emb, run_time=cycle_time))

        ## Let embedding flow
        self.remove(stream_lines)
        self.add(stream_lines_emb)
        stream_lines_emb.start_animation(warm_up=False, flow_speed=1)
        self.wait(cycle_time)

        ## Remove visual aid
        self.play(FadeOut(embedding_surface), run_time=cycle_time)

        ## Show stream lines alone
        self.wait(cycle_time)

        ## TODO: Show plane
        #self.begin_ambient_camera_rotation(rate=-0.2)
        #self.wait(cycle_time)
        #self.play(FadeIn(linear_surface), run_time=cycle_time)
        self.move_camera(phi=40.8 * DEGREES, theta=90 * DEGREES,frame_center=[0,0,0], run_time=cycle_time)
        #self.wait(cycle_time * 4.55)
        #self.stop_ambient_camera_rotation()
        self.wait(cycle_time)



        #self.add(surface) 
        #self.add(stream_lines_emb)
        # animate camera movement
        #stream_lines.start_animation(warm_up=False, flow_speed=1)
        #self.move_camera(phi=45 * DEGREES, theta=-90 * DEGREES,frame_center=[0,0,1], run_time=3)
        #stream_lines_emb.start_animation(warm_up=False, flow_speed=1)
        #self.begin_ambient_camera_rotation(rate=0.2)
        #self.wait(5)
        #stream_lines.fit_to_coordinate_system(axes)
        #self.add(stream_lines)
        #stream_lines.start_animation(warm_up=False, flow_speed=1)
        #self.wait(stream_lines.virtual_time / stream_lines.flow_speed)
        #self.remove(stream_lines)

        #stream_lines_emb.fit_to_coordinate_system(axes)
        #self.set_camera_orientation(phi=75 * DEGREES, theta=70 * DEGREES, frame_center=[0,0,3])
        #self.add(stream_lines_emb)
        #stream_lines_emb.start_animation(warm_up=False, flow_speed=1)
        #self.wait(stream_lines_emb.virtual_time / stream_lines_emb.flow_speed)

        #self.wait(stream_lines.virtual_time / stream_lines.flow_speed)
        #stream_lines.end_animation()
        #self.play(Transform(stream_lines, stream_lines_emb, run_time=3))

