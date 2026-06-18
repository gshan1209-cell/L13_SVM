from manim import *
import numpy as np
import sys
import os

# Ensure the parent directory is in the path so we can import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.data_generator import generate_ring_dataset

class SVMKernelTrick3D(ThreeDScene):
    def construct(self):
        # 1. Opening Title
        title = Text("SVM Kernel Trick: From 2D to 3D", font_size=36, color=WHITE)
        subtitle = Text("Nonlinear in 2D, linear in feature space.", font_size=20, color=GRAY)
        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.3)
        title_group.to_edge(UP)
        
        self.add_fixed_in_frame_mobjects(title_group)
        self.play(FadeIn(title_group))
        self.wait(1.5)
        
        # 2. Show 2D Data
        # Set up 3D Axes
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[0, 8, 2],
            x_length=6,
            y_length=6,
            z_length=4
        )
        self.set_camera_orientation(phi=0 * DEGREES, theta=-90 * DEGREES) # Look from top (2D view)
        
        # Generate ring data
        X, y = generate_ring_dataset(
            n_inner=35,
            n_outer=45,
            inner_radius_range=(0.0, 1.0),
            outer_radius_range=(1.6, 2.5),
            noise=0.0,  # Pure Concentric rings for clean mathematical visualization
            random_seed=7
        )
        
        # Create dots at z=0
        blue_dots = VGroup()
        red_dots = VGroup()
        
        for i, (x_val, y_val) in enumerate(X):
            dot = Dot3D(
                point=axes.c2p(x_val, y_val, 0),
                color=BLUE if y[i] == 0 else RED,
                radius=0.08
            )
            if y[i] == 0:
                blue_dots.add(dot)
            else:
                red_dots.add(dot)
                
        data_dots = VGroup(blue_dots, red_dots)
        
        self.play(FadeIn(axes), FadeIn(data_dots))
        self.wait(1)
        
        # Explain 2D nonlinear
        desc_2d = Text("No straight line can separate them in 2D.", font_size=20, color=YELLOW)
        desc_2d.next_to(title_group, DOWN, buff=0.2)
        self.add_fixed_in_frame_mobjects(desc_2d)
        self.play(Write(desc_2d))
        self.wait(2)
        self.play(FadeOut(desc_2d))
        
        # 3. Show Mapping Formula
        formula = MathTex(
            r"\phi(x, y) = (x, y, x^2 + y^2)",
            font_size=32,
            color=YELLOW
        )
        formula.next_to(title_group, DOWN, buff=0.2)
        self.add_fixed_in_frame_mobjects(formula)
        self.play(Write(formula))
        self.wait(1.5)
        
        # 4. Camera rotation to 3D orientation & Lift points to 3D
        self.move_camera(phi=65 * DEGREES, theta=-45 * DEGREES, run_time=2)
        self.wait(0.5)
        
        # Lift dots
        lifting_animations = []
        for i, dot in enumerate(blue_dots):
            x_val, y_val = X[y == 0][i]
            z_val = x_val**2 + y_val**2
            lifting_animations.append(
                dot.animate.move_to(axes.c2p(x_val, y_val, z_val))
            )
        for i, dot in enumerate(red_dots):
            x_val, y_val = X[y == 1][i]
            z_val = x_val**2 + y_val**2
            lifting_animations.append(
                dot.animate.move_to(axes.c2p(x_val, y_val, z_val))
            )
            
        self.play(*lifting_animations, run_time=2.5)
        self.wait(1)
        
        # 5. Show Paraboloid Surface
        paraboloid = Surface(
            lambda u, v: axes.c2p(u, v, u**2 + v**2),
            u_range=[-2.5, 2.5],
            v_range=[-2.5, 2.5],
            resolution=(25, 25),
            fill_opacity=0.25,
            checkerboard_colors=[BLUE_D, BLUE_E]
        )
        self.play(Create(paraboloid), run_time=2)
        self.wait(1)
        
        # 6. Show Separating Hyperplane
        c = 1.8  # Height chosen to lie between inner circle (z<=1.0) and outer circle (z>=2.56)
        hyperplane = Surface(
            lambda u, v: axes.c2p(u, v, c),
            u_range=[-2.5, 2.5],
            v_range=[-2.5, 2.5],
            resolution=(10, 10),
            fill_opacity=0.45,
            checkerboard_colors=[YELLOW_D, YELLOW_E]
        )
        plane_label = Text("Hyperplane in feature space (z = c)", font_size=16, color=YELLOW)
        plane_label.next_to(formula, DOWN, buff=0.2)
        self.add_fixed_in_frame_mobjects(plane_label)
        
        self.play(Create(hyperplane), Write(plane_label), run_time=2)
        self.wait(1.5)
        
        # 7. Project Back to 2D
        r_boundary = np.sqrt(c)
        decision_boundary_3d = ParametricFunction(
            lambda t: axes.c2p(r_boundary * np.cos(t), r_boundary * np.sin(t), c),
            t_range=[0, 2 * np.pi],
            color=YELLOW,
            stroke_width=6
        )
        decision_boundary_2d = ParametricFunction(
            lambda t: axes.c2p(r_boundary * np.cos(t), r_boundary * np.sin(t), 0),
            t_range=[0, 2 * np.pi],
            color=YELLOW,
            stroke_width=4
        )
        
        self.play(Create(decision_boundary_3d))
        self.wait(1)
        self.play(Transform(decision_boundary_3d, decision_boundary_2d))
        self.wait(1)
        
        # 8. Camera Rotation
        self.begin_ambient_camera_rotation(rate=0.18)
        self.wait(4)
        self.stop_ambient_camera_rotation()
        self.wait(1)
        
        # 9. Final Summary
        self.play(
            FadeOut(paraboloid),
            FadeOut(hyperplane),
            FadeOut(plane_label)
        )
        
        summary_title = Text("Key Takeaways", font_size=24, color=YELLOW)
        summary_title.next_to(formula, DOWN, buff=0.3)
        
        takeaway1 = Text("- In 3D: linear separating hyperplane", font_size=18, color=WHITE)
        takeaway2 = Text("- In 2D: nonlinear circular decision boundary", font_size=18, color=WHITE)
        takeaway3 = Text("- This is the intuition behind the Kernel Trick", font_size=18, color=WHITE)
        
        summary_group = VGroup(summary_title, takeaway1, takeaway2, takeaway3).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        summary_group.next_to(formula, DOWN, buff=0.4)
        self.add_fixed_in_frame_mobjects(summary_group)
        
        self.play(Write(summary_group))
        self.wait(4)
        
        # Outro Fade
        self.play(
            FadeOut(title_group),
            FadeOut(formula),
            FadeOut(summary_group),
            FadeOut(axes),
            FadeOut(data_dots),
            FadeOut(decision_boundary_3d)
        )
        self.wait(1)
