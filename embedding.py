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

class Intro(ThreeDScene):
    def construct(self):
        spacing = 1.25
        welcome = Text("Intro to:", font_size=72).to_edge(UP, buff=spacing)
        koopman = MathTex(r"\mathcal{K}", font_size=2*72).next_to(welcome, DOWN, buff=spacing)
        title = Text("The Koopman Operator", font_size=72).next_to(koopman, DOWN, buff=spacing)

        self.add(welcome)
        self.add(koopman)
        self.add(title)

class Derivation(ThreeDScene):
    def construct(self):
        spacing = 0.5
        ode = MathTex(r"\frac{dx}{dt}=f(x, t)", font_size=72).to_edge(UP, buff=spacing)
        discrete_ode = MathTex(r"x_{k+1}=F(x_{k})", font_size=72).next_to(ode, DOWN, buff=spacing)
        embedding = MathTex(r"\phi(x_{k+1})=\phi(F(x_k))", font_size=72).next_to(discrete_ode, DOWN, buff=spacing)
        koopman = MathTex(r"\phi(x_{k+1})=\mathcal{K}\phi(x_k)", font_size=72).next_to(embedding, DOWN, buff=spacing)

        self.add(ode)
        self.play(TransformFromCopy(ode, discrete_ode, run_time=2))
        self.play(TransformFromCopy(discrete_ode, embedding, run_time=2))
        self.play(TransformFromCopy(embedding, koopman, run_time=2))

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

