import numpy as np
from scipy.integrate import solve_ivp
from manim import *
from manim_mobject_svg import *

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

FORMULA_BACK = '#D2D2FF'
FORMULA_FS = 24
FORMULA_TC = '#FFFFFF'

def make_formula(*obs, width=None):
    group = VGroup([ob for ob in obs]).arrange(DOWN, buff=0.2)
    if width is not None:
        print(width)
        group_rect = BackgroundRectangle(group, corner_radius=0.2, color=FORMULA_BACK, fill_opacity=0.5, buff=0.1, width=width)
    else:
        group_rect = BackgroundRectangle(group,corner_radius=0.2, color=FORMULA_BACK, fill_opacity=0.5, buff=0.1)
    group_rect.set_background_stroke(width=2, color=FORMULA_TC)
    group.add(group_rect)
    return group

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

class Motivation(ThreeDScene):
    def construct(self):
        return


class Derivation(ThreeDScene):
    def construct(self):
        template = TexTemplate()
        template.add_to_preamble(r"\usepackage{mathtools}")

        elements = VGroup()


        # Full aligned block
        assumptions_desc = Tex("We look at autonomous discrete ODEs with finite state space:", font_size=FORMULA_FS)
        assumptions_math = MathTex(
            r"&{\bf x}\in\mathcal{X}, \mathcal{X}\subseteq\mathbb{R}^n, t\in \mathbb{N} \\",
        )
        assumptions = make_formula(assumptions_desc, assumptions_math).to_corner(UL, buff=0.5)
        elements.add(assumptions)


        flow_map_desc = Tex("So our state is advanced by a \\textit{flow map} ${\\bf F}$:", font_size=FORMULA_FS)
        flow_map_math = MathTex(
            r"{\bf x}_{k+1} = {\bf F}({\bf x}_k), \quad {\bf F}:\mathcal{X}\to\mathcal{X}"
        )
        flow_map = make_formula(flow_map_desc, flow_map_math)
        flow_map[-1].stretch_to_fit_width(assumptions[-1].get_width())
        flow_map.next_to(assumptions, DOWN, aligned_edge=LEFT, buff=0.5)
        elements.add(flow_map)

        goal_desc = Tex("Ideally, we want to find coordinates ${\\bf z}$ such that", font_size=FORMULA_FS)
        goal_math = MathTex(
            r"{\bf z}_{k+1} = {\bf K}{\bf z}_k"
        )
        goal_comment = Tex("Where ${\\bf K}$ is a matrix", font_size=FORMULA_FS)
        goal = make_formula(goal_desc, goal_math, goal_comment)
        goal[-1].stretch_to_fit_width(assumptions[-1].get_width())
        goal.align_to(LEFT).to_corner(DL, buff=0.5)

        elements.add(goal)

        coord_desc = Tex("The elements of ${\\bf z_i}$ should be a function of our state vector", font_size=FORMULA_FS).to_edge(DOWN, buff=-0.5)
        elements.add(coord_desc)

        m_space_desc = Tex("To make this general, we define a set of \\textit{measurement functions}:", font_size=FORMULA_FS)
        m_space_math = MathTex(
            r"{\cal G}(\mathcal{X})=\{g\colon\mathcal{X}\to\mathbb{C}\}, g\in{\cal G}(\mathcal{X})"
        )
        m_space_comment = Tex("We choose it to be a linear vector space, e.g. ${\\bf L}^2$", font_size=FORMULA_FS)
        m_space = make_formula(m_space_desc, m_space_math, m_space_comment)
        m_space.align_to(LEFT).to_corner(UR, buff=0.5)
        elements.add(m_space)

        def_koop_desc = Tex("Now we can define an operator\\\\ that advances measurement functions by one timestep:", font_size=FORMULA_FS)
        def_koop_math = MathTex(r"\mathcal{K}\colon{\cal G}({\cal X})\to{\cal G}({\cal X}) \\",
                                r"\mathcal{K}g({\bf x})=g({\bf F}({\bf x}))")
        def_koop_comment = Tex("${\\cal K}$ is called the \\textit{Koopman operator}", font_size=FORMULA_FS)
        def_koop = make_formula(def_koop_desc, def_koop_math, def_koop_comment)
        def_koop[-1].stretch_to_fit_width(m_space[-1].get_width())
        def_koop.align_to(LEFT).next_to(m_space, DOWN, aligned_edge=RIGHT, buff=0.5)
        elements.add(def_koop)

        # align flow map
        flow_map.align_to(def_koop, UP)

        goal_koop_desc = Tex("With a bit of different notation:", font_size=FORMULA_FS)
        goal_koop_math = MathTex(
            r"{\bf g}_{k+1} = {\cal K}{\bf g}_k"
        )
        goal_koop_comment = Tex("We are pretty close to what we want!", font_size=FORMULA_FS)
        goal_koop = make_formula(goal_koop_desc, goal_koop_math, goal_koop_comment)
        goal_koop[-1].stretch_to_fit_width(m_space[-1].get_width())
        goal_koop.align_to(LEFT).to_corner(DR, buff=0.5)
        elements.add(goal_koop)

        elements.to_svg("derivation.svg")
        self.add(elements)


class EigenfunctionsDef(ThreeDScene):
    def construct(self):
        prelude = Tex(r"Now that we are in a linear setting, we linear analysis tools!", color=FORMULA_TC).to_edge(UP, buff=0.5)

        comparision = Tex(r"Similar to eigenvectors for matricies, \\" 
            r"there are \textit{eigenfunctions} $\varphi({\bf x})$ of $\mathcal{K}$", font_size=28, color=FORMULA_TC)
        def_eig_fun = MathTex(r"\varphi({\bf x}_{k+1})={\cal K}\varphi({\bf x}_{k})=\lambda\varphi({\bf x}_{k}).")
        eig_val = Tex(r"with corresponding \textit{eigenvalues} $\lambda\in\mathbb{C}$", font_size=28, color=FORMULA_TC)

        eig_fun = make_formula(
            comparision, def_eig_fun, eig_val,
        )
        self.add(prelude, eig_fun)

        expl = Tex(
            r"These functions are time-invariant directions in our observable space $\mathcal{G}(\mathcal{X})$"
            ).to_edge(DOWN, buff=0.5)

#        self.play(Write(comparision))
#        self.play(Write(def_eig_fun))
#        self.play(Write(expl))
#
#        self.play(FadeOut(comparision))
#        self.play(def_eig_fun.animate.to_edge(UP),
#        expl.animate.to_edge(UP, buff=2))
        self.add(comparision, def_eig_fun, expl)
        
        exp_group = VGroup(comparision, def_eig_fun, expl)
        exp_group.to_svg("eigenfunctions_def.svg")

class Eigenfunctions(ThreeDScene):
    def construct(self):
        title = Tex("Some important properties of Eigenfunctions:", font_size=36).to_edge(UP, buff=0.5)
        self.add(title.to_corner(UL))

        multipl_math = MathTex(
            r"{\cal K}(\varphi_{1}({\bf x})\varphi_{2}({\bf x}))&=\varphi_{1}({\bf F}({\bf x}))\varphi_{2}({\bf F}({\bf x}))\\",
            r"&=\lambda_{1}\lambda_{2}\varphi_{1}({\bf x})\varphi_{2}({\bf x})"
        )
        multipl_text = Tex(
            r"The product of two Eigenfunctions is an Eigenfunctions \\",
            r"(if ${\cal G}({\cal X})$ is closed under multiplication)", font_size=FORMULA_FS, color=FORMULA_TC
        )
        multipl = make_formula(multipl_text, multipl_math).to_edge(UP, buff=1.5)

        evolve_math = MathTex(r"g({\bf x})=\sum_{k}v_{k}\varphi_{k}\quad\Longrightarrow\quad {\cal K}^{t}g({\bf x})=\sum_{k}v_{k}\lambda_{k}^{t}\varphi_{k}")
        evolve_text = Tex(
            r"Observables $g\in span\{\varphi_k\}^K_{k=1}$ evolve particulary simple",font_size=FORMULA_FS, color=FORMULA_TC)
        evolve_subtext = Tex(r"So $span\{\varphi_k\}^K_{k=1}\subseteq{\cal G}({\cal X})$ is invariant under the action of ${\cal K}$", font_size=FORMULA_FS, color=FORMULA_TC)
        evolve = make_formula(evolve_text, evolve_math, evolve_subtext).next_to(multipl, DOWN, buff=1.5)

        multipl[-1].set_width(evolve[-1].get_width())
        self.add(multipl)
        self.add(evolve)

class KMD(ThreeDScene):
    def construct(self):
        single_text = Tex("Instead of one we look at multiple measurements $g_i$\\\\which decompose as follows:", font_size=FORMULA_FS, color=FORMULA_TC)
        single_math = MathTex(
            r"g_{i}({\bf x})=\sum_{j=1}^{\infty}v_{i j}\varphi_{j}({\bf x})"
        )
        single = make_formula(single_text, single_math).to_corner(UL, buff=0.5)

        measurements_text = Tex("So we can write our measurements as a vector ${\\bf g}({\\bf x})$:", font_size=FORMULA_FS, color=FORMULA_TC)
        measurements_vec = Matrix([[r"g_1({\bf x})"], [r"g_2({\bf x})"], [r"\vdots"],[r"g_p({\bf x})"]], h_buff=0.5, element_alignment_corner=[0, 0, 0])
        measurements_exp = MathTex(
            r"=\sum_{j=1}^{\infty}\varphi_{j}(\mathbf{x})\mathbf{v}_{j}"
        )
        measurements_math = VGroup(measurements_vec, measurements_exp).arrange(RIGHT)
        measurements = make_formula(measurements_text, measurements_math).next_to(single, DOWN, aligned_edge=LEFT, buff=0.5)

        single[-1].stretch_to_fit_width(measurements[-1].get_width())

        self.add(measurements)

        repr_text = Tex("And the evolution of our measurements can be easily calculated using the Eigenvalues:", font_size=FORMULA_FS, color=FORMULA_TC)
        repr_math= MathTex(
            r"{\bf g}({\bf x}(t))={\cal K}^t{\bf g}({\bf x}_0)&={\cal K}^t\sum_{j=1}^{\infty}\varphi_j({\bf x}_0){\bf v}_j\\",
            r"&=\sum_{j=1}^{\infty}{\cal K}^{t}\varphi_{j}({\bf x}_{0}){\bf v}_j\\ ",
            r"&=\sum_{j=1}^{\infty}\lambda_{j}^{t}\varphi_{j}({\bf x}_{0}){\bf v}_j",
        )
        repr = make_formula(repr_text, repr_math).to_corner(UR, buff=0.5)

        kmd_text = Tex("The sequence of triplets $\\{(\\lambda_j,\\varphi_j,{\\bf v}_j)\\}^\\infty_{j=1}$ is the\\\\\\textit{Koopman mode decomposition}", font_size=FORMULA_FS, color=FORMULA_TC)
        kmd = make_formula(kmd_text)
        kmd[-1].stretch_to_fit_width(repr[-1].get_width())
        kmd.next_to(repr, DOWN, buff=0.5)
        kmd.align_to(measurements, DOWN)

        comment = Tex("Of course, infinite sums are not practical,\\\\so we have to somehow restrict the Koopman Operator", font_size=FORMULA_FS, color=FORMULA_TC)
        comment.to_edge(DOWN, buff=-0.5)







        self.add(measurements, repr, single, comment, kmd)

class InvariantSubspaceStatic(ThreeDScene):
    def construct(self):
        inv_sub_def = Tex("An \\textit{invariant subspace} of ${\\cal K}$ is a subspace spanned by a set of functions $\\{g_1, g_2, \\ldots, g_p\\}$ such that", font_size=FORMULA_FS, color=FORMULA_TC)
        inv_sub_math = MathTex(
            r"{g=\sum_{i=1}^{p}{\alpha_i g_i}}\Longrightarrow{\cal K}g=\sum_{i=1}^{p}{\beta_i g_i} \quad\quad \alpha_i,\beta_i\in\mathcal{X}",
        )
        inv_sub_desc = Tex("all linear combinations of $g_i$ stay in the space under the action of ${\\cal K}$", font_size=FORMULA_FS, color=FORMULA_TC)
        inv_sub = make_formula(inv_sub_def, inv_sub_math, inv_sub_desc).to_edge(UP, buff=0.0)
        self.add(inv_sub)

        koop_mat_text = Tex("If we restrict ${\\cal K}$ to such an invariant subspace, we can represent it as a matrix:", font_size=FORMULA_FS, color=FORMULA_TC)
        koop_mat_math = MathTex(r"{\cal K}={\bf K}", font_size=72)
        koop_mat_sub = Tex("and it acts on a vector space $\\mathbb{R}^p$, with coordinates by the values of $g_j({\\bf x})$, resulting in a finite linear system", font_size=FORMULA_FS, color=FORMULA_TC)
        koop_mat = make_formula(koop_mat_text, koop_mat_math, koop_mat_sub).next_to(inv_sub,DOWN, buff=1.0)
        self.add(koop_mat)

        koop_mat[-1].stretch_to_fit_width(inv_sub[-1].get_width())

        eig_span_text = Tex("Actually any finite set of eigenfunctions of ${\\cal K}$\\\\spans an invariant subspace of ${\\cal G}({\\cal X})$!", color=FORMULA_TC)
        eig_span_text.to_edge(DOWN, buff=-0.5)
        self.add(eig_span_text)



class InvariantSubspace(ThreeDScene):
    def construct(self):
        self.next_section("title")

        title = Tex("Invariant Subspaces", font_size=72).to_edge(UP, buff=0.5)
        self.add(title)
        span = Tex(
            r"$span({g_1, g_2, \ldots, g_p})\subseteq{\cal G}({\cal X}), p\in\mathbb{N}$ \\",
        ).next_to(title, DOWN, buff=0.5)
        subspace = Tex(
            r"where all linear combinations are invariant under ${\cal K}$:"
        ).next_to(span, DOWN, buff=0.5)
        formula = MathTex(
            r"{g=\sum_{i=1}^{p}{\alpha_i g_i}}\Longrightarrow{\cal K}g=\sum_{i=1}^{p}{\beta_i g_i} \quad\quad \alpha_i,\beta_i\in\mathcal{X}",
        ).next_to(span, DOWN, buff=0.5)
        span.set_color_by_tex("G", YELLOW_C)

        koopman_matrix = Tex(
            r"If we restict ${\cal K}$ to such an invariant subspace, \\"
            r"we can represent it as a matrix:"
        ).next_to(formula, DOWN, buff=0.5)

        symbol_operator = MathTex(r"{\cal K}", font_size=72).next_to(koopman_matrix, DOWN, buff=0.5)
        symbol_matrix= MathTex(r"{\bf K}", font_size=72).next_to(koopman_matrix, DOWN, buff=0.5)
        self.add(span, subspace)

        self.next_section("transform_formula")

        self.play(Transform(subspace, formula, run_time=2, fade_transform_mismatches=True))

        self.next_section("koopman_matrix")

        self.play(Write(koopman_matrix))

        self.next_section("symbol_operator")

        self.play(TransformMatchingTex(symbol_operator, symbol_matrix, run_time=2, fade_transform_mismatches=True))

        self.play(FadeOut(span), FadeOut(subspace), FadeOut(formula), FadeOut(koopman_matrix))

        self.play(symbol_matrix.animate.next_to(title, DOWN, buff=0.5))
        self.play(symbol_matrix.animate.set_color(YELLOW_C))

        vec_space = Tex(
            r"acts on a vector space $\mathbb{K}^p$, \\"
            r"with coordinates by the values of $g_j({\bf x})$, \\"
            r"resulting in a finite linear system"
        ).next_to(symbol_matrix, DOWN, buff=0.5)
        self.play(Write(vec_space))

        eig_span_invariant = Tex(
            r"Any finite set of eigenfunctions of ${\cal K}$ \\"
            r"spans an invariant subspace of ${\cal G}({\cal X})$!"
        ).next_to(vec_space, DOWN, buff=0.5)
        eig_span_invariant.set_color_by_tex("K", RED_Z_1)

        self.play(Write(eig_span_invariant))

class AbstractEmbedding(ThreeDScene):
    def construct(self):
        state_descr = VGroup([Tex("Finite", font_size=72),Tex("nonlinear", font_size=72),Tex("state"," space", font_size=72)])
        state_descr.arrange(DOWN, buff=0.25).to_corner(UL, buff=0.5)
        state_descr[0].set_color(GREEN)
        state_descr[1].set_color(RED)
        state_descr[2][0].set_color(YELLOW_C)
        x = MathTex(r"x \in \mathcal{X}", font_size=72).next_to(state_descr, DOWN, buff=0.5)
        arr = Arrow(buff=0.5)
        state_vec = Matrix([["x_1"], ["x_2"], ["\\vdots"],["x_n"]], h_buff=0.5, element_alignment_corner=[0, 0, 0]).next_to(x, DOWN, buff=0.5)
        self.add(state_descr, x, arr, state_vec)

        state_descr2 = state_descr.copy()
        x2 = x.copy()
        state_vec2 = state_vec.copy()

        observ_descr = VGroup([Tex("Infinite", font_size=72),Tex("linear", font_size=72),Tex("observable"," space", font_size=72)])
        observ_descr.arrange(DOWN, buff=0.25).to_corner(UR, buff=0.5)
        observ_descr[0].set_color(RED)
        observ_descr[1].set_color(GREEN)
        observ_descr[2][0].set_color(PURPLE_A)
        phi = MathTex(r"\Phi \in {\cal G}({\cal X})^\infty", font_size=72).next_to(observ_descr, DOWN, buff=0.5)
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

class DMD(ThreeDScene):
    def construct(self):
#        dmd_desc = Tex(r"\textit{Dynamic Mode Decomposition} (DMD) is a data-driven method, \\"
#                       r"which finds a matrix approximation of the Koopman operator ${\cal K}$, \\"
#                       r"restricted to the subspace spaned by the state variables themselves \\"
#                       , font_size=FORMULA_FS, color=FORMULA_TC)
        dmd_title = Tex(r"\textit{Dynamic Mode Decomposition} (DMD):",
                       font_size=FORMULA_FS, color=FORMULA_TC)
        dmd_props = BulletedList(
            r"Data-driven method",
            r"Finds a matrix approximation of the Koopman operator ${\cal K}$",
            r"${\cal K}$ is restricted to the subspace spanned by the state variables"
        , font_size=FORMULA_FS, color=FORMULA_TC, buff=0.2)
        dmd_math = MathTex(
            r"x_{k+1} = {\bf A}x_k"
        )
        dmd = make_formula(dmd_title, dmd_props, dmd_math)
        dmd.to_corner(UL, buff=0.5)
        self.add(dmd)
        #self.wait(1)

        self.next_section("dmd_eq")

        #self.play(TransformMatchingTex(dmd_desc, dmd_eq, run_time=2, fade_transform_mismatches=True))

        MATRIX_FS = 36
        snapshots_text = Tex(r"Input: snapshots of the state vector ${\bf x}$ at arbitrary time steps:", font_size=FORMULA_FS, color=FORMULA_TC)
        snapshots = VGroup(
            MathTex(r"{\bf X}\;=", font_size=MATRIX_FS),
            Matrix([["|", "|", "", "|"], ["{\\bf x}(t_1)", "{\\bf x}(t_2)", "\cdots", "{\\bf x}(t_m)"], ["|", "|", "", "|"]], h_buff=1.0, v_buff=0.5, element_alignment_corner=[0, 0, 0],
                   element_to_mobject_config={"font_size": MATRIX_FS}),
        ).arrange(RIGHT, buff=0.25)
        snapshots_delay_text = Tex(
            r"and of corresponding subsequent time step ${t_i^\prime}=t_i + \Delta t$:", font_size=FORMULA_FS, color=FORMULA_TC)
        snapshots_delay = VGroup(
            MathTex(r"{\bf X^\prime}=", font_size=MATRIX_FS),
            Matrix([["|", "|", "", "|"], ["{\\bf x}(t^\prime_1)", "{\\bf x}(t^\prime_2)", "\cdots", "{\\bf x}(t^\prime_m)"], ["|", "|", "", "|"]], h_buff=1.0, v_buff=0.5, element_alignment_corner=[0, 0, 0],
                   element_to_mobject_config={"font_size": MATRIX_FS}),
        ).arrange(RIGHT, buff=0.25)

        print(snapshots_delay[0].font_size)
        snaps = make_formula(snapshots_text, snapshots, snapshots_delay_text, snapshots_delay)
        snaps.next_to(dmd, DOWN, buff=0.5)

        snaps[-1].stretch_to_fit_width(dmd[-1].get_width())
        self.add(snaps)

        opti_text = Tex("We need to solve the following optimization problem:", font_size=FORMULA_FS, color=FORMULA_TC)
        approx_math = MathTex(
            r"{\bf X^\prime}\approx{\bf AX"
        )
        opti_math = MathTex(
            r"\mathbf{A}=\operatorname*{argmin}_{\mathbf{A}}\|\mathbf{X}^{\prime}-\mathbf{A}\mathbf{X}\|_{F}=\mathbf{X}^{\prime}\mathbf{X}^{\dagger"
        )
        opti = make_formula(opti_text, approx_math, opti_math)
        opti.to_corner(UR, buff=0.5)

        eig_vecs_text = Tex("We focus on its Eigenvectors and Eigenvalues", font_size=FORMULA_FS, color=FORMULA_TC)
        eig_vecs_math = MathTex(
            r"{\bf A\Phi} = {\bf \Phi\Lambda}"
        )

        svd_text = Tex("Using the SVD of our snapshots matrix ${\\bf X}$:", font_size=FORMULA_FS, color=FORMULA_TC)
        svd = MathTex(
            r"{\bf X} = {\bf U\Sigma V}^*\Rightarrow {\bf X}^\dagger = {\bf V\Sigma}^{-1}{\bf U}^*"
        )
        svd_subtext = Tex("Now, same idea as for  SVD, calculate smaller ${\\bf \\tilde{A} }$\\\\by projection on r largest singular values", font_size=FORMULA_FS, color=FORMULA_TC)
        svd = make_formula(eig_vecs_text, eig_vecs_math, svd_text, svd, svd_subtext)
        svd.next_to(opti, DOWN, aligned_edge=RIGHT, buff=0.5)

        self.add(opti, svd)

        decomp_text = Tex("We can now represent our system state in terms of the DMD expansion:", font_size=FORMULA_FS, color=FORMULA_TC)
        decomp_math = MathTex(
            r"{\bf x}_{k}=\sum_{j=1}^{r}\phi_{j}\lambda_{j}^{k-1}b_{j}=\Phi{\bf A}^{k-1}{\bf b}"
        )
        decomp = make_formula(decomp_text, decomp_math)
        decomp.next_to(svd, DOWN, aligned_edge=RIGHT, buff=0.5)
        self.add(decomp)
        svd[-1].stretch_to_fit_width(decomp[-1].get_width())
        opti[-1].stretch_to_fit_width(decomp[-1].get_width())
        snaps.align_to(decomp, DOWN)
        opti.align_to(decomp, LEFT)
        svd.align_to(decomp, LEFT)

        #self.next_section("dmd_matrix")

        #dmd_matrix = MathTex(
        #    r"{\bf K}={\bf V}{\bf \Lambda}{\bf V}^{-1}"
        #).next_to(dmd_eq, DOWN, buff=0.5)
        #dmd_matrix.set_color_by_tex("K", YELLOW_C)
        #self.add(dmd_matrix)

class DMDExpansion(ThreeDScene):
    def construct(self):
        return

class SimpleEmbedding(ThreeDScene):
    def observable(self, u, v):
       return np.array([u, v, u**2])

    def construct(self):
        ### SETUP ###
        axes = ThreeDAxes(z_range=[0, 1, 1], x_range=[-1, 1, 1], y_range=[-1, 1, 1])
        axes.set_color(GREY)
        axes.set_stroke(width=0.5)
        axes.set_opacity(0.5)
        mu = -0.05
        l = -1

        system = lambda y: np.array([mu*y[0], l * (y[1] - y[0]**2), 0])

        stream_lines = StreamLines(
            system,
            x_range=[-1, 1, 0.1],
            y_range=[-1, 1, 0.1],
            stroke_width=2,
            max_anchors_per_line=100,
            padding=1,
            virtual_time=5,
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
        stream_lines.start_animation(warm_up=True, flow_speed=1, time_width=1.0)
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
        self.begin_ambient_camera_rotation(rate=-0.2)
        self.wait(cycle_time)
        self.play(FadeIn(linear_surface), run_time=cycle_time)
        self.move_camera(phi=40.8 * DEGREES, theta=90 * DEGREES,frame_center=[0,0,0], run_time=cycle_time)
        self.wait(cycle_time * 4.55)
        self.stop_ambient_camera_rotation()
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


class SimpleEmbeddingEig(ThreeDScene):
    def observable(self, u, v):
       return np.array([u, v, u**2])

    def construct(self):
        ### SETUP ###
        axes = ThreeDAxes(z_range=[0, 1, 1], x_range=[-1, 1, 1], y_range=[-1, 1, 1])
        axes.set_color(GREY)
        axes.set_stroke(width=0.5)
        axes.set_opacity(0.5)
        mu = -0.05
        l = -1

        self.camera.set_focal_distance(1000)

        system = lambda y: np.array([mu*y[0], l * (y[1] - y[0]**2), 0])

        def speed(v):
            return np.linalg.norm(v)

        stream_lines = StreamLines(
            system,
            x_range=[-1, 1, 0.1],
            y_range=[-1, 1, 0.1],
            stroke_width=2,
            max_anchors_per_line=100,
            padding=1,
            virtual_time=5,
            colors=[BLUE_Z_2, BLUE_Z_1],
            color_scheme=speed,
            min_color_scheme_value=0.0,
            max_color_scheme_value=1.0,
        )

        eig_fun1 = lambda y: y[0]
        eig_fun2 = lambda y: y[1] - (l/(l-2*mu)) * y[0]**2

        stream_lines_eig = stream_lines.copy()
        for line in stream_lines_eig:
            for point in line.points:
                point[0] = eig_fun1(point[:2])
                point[1] = eig_fun2(point[:2])



        stream_lines.fit_to_coordinate_system(axes)
        stream_lines_eig.fit_to_coordinate_system(axes)

        # time for stream lines to complete one cycle
        cycle_time = stream_lines.virtual_time
        ### ANIMATION ###

        self.add(axes)

        ## Show nonlinear system
        '''
        self.add(stream_lines)
        stream_lines.start_animation(warm_up=True, flow_speed=1, time_width=1.0)
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
        self.move_camera(phi=40.8 * DEGREES, theta=90 * DEGREES,frame_center=[0,0,0], run_time=cycle_time)
        self.wait(cycle_time)
        '''
        #self.add(stream_lines)

        x = np.linspace(-1, 1, 100)
        y = np.linspace(-1, 1, 100)
        z = eig_fun2([x, y])

        graph = axes.plot_line_graph(
            x, y, z, line_color=RED_Z_1, stroke_width=2
        )
        #self.add(stream_lines_emb, graph)
        #self.set_camera_orientation(phi=39 * DEGREES, theta=-45 * DEGREES,frame_center=[0,0,0])
        #self.begin_ambient_camera_rotation(rate=0.1)

        #self.wait(10)
        self.next_section("original")
        self.add(stream_lines)
        stream_lines.start_animation(warm_up=True, flow_speed=1, time_width=1.0)
        self.wait(cycle_time)
        stream_lines.end_animation()
        self.next_section("transform")
        self.play(Transform(stream_lines, stream_lines_eig, run_time=cycle_time))
        self.next_section("linear")
        self.remove(stream_lines)
        self.add(stream_lines_eig)
        stream_lines_eig.start_animation(warm_up=False, flow_speed=1, time_width=1.0)
        self.wait(cycle_time)

        #self.move_camera(
        #    phi=48.01 * DEGREES, theta = -90* DEGREES, frame_center=[0, 0, 0], run_time = 2
        #)