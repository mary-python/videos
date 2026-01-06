from manimlib import *
import numpy as np


if TYPE_CHECKING:
    from typing import Callable, Iterable, Sequence, TypeVar, Tuple, Optional
    from manimlib.typing import Vect2, Vect3, VectN, VectArray, Vect2Array, Vect3Array, Vect4Array


def fibonacci_sphere(samples=1000):
    """
    Create uniform-ish points on a sphere

    Parameters
    ----------
    samples : int
        Number of points to create. The default is 1000.

    Returns
    -------
    points : NumPy array
        Points on the unit sphere.

    """

    # Define the golden angle
    phi = np.pi * (np.sqrt(5) - 1)

    # Define y-values of points
    pos = np.array(range(samples), ndmin=2)
    y = 1 - (pos / (samples - 1)) * 2

    # Define radius of cross-section at y
    radius = np.sqrt(1 - y * y)

    # Define the golden angle increment
    theta = phi * pos

    # Define x- and z- values of poitns
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    # Merge together x,y,z
    points = np.concatenate((x, y, z))

    # Transpose to get coordinates in right place
    points = np.transpose(points)

    return points


def stereographic_proj(points3d, epsilon=1e-10):
    x, y, z = points3d.T

    denom = 1 - z
    denom[np.abs(denom) < epsilon] = np.inf
    return np.array([x / denom, y / denom, 0 * z]).T


def inv_streographic_proj(points2d):
    u, v = points2d.T
    norm_squared = u * u + v * v
    denom = 1 + norm_squared
    return np.array([
        2 * u / denom,
        2 * v / denom,
        (norm_squared - 1) / denom,
    ]).T


def right_func(points):
    return np.repeat([[1, 0]], len(points), axis=0)


def stereographic_vector_field(points3d, vector_field_2d):
    points2d = stereographic_proj(points3d)[:, :2]
    vects2d = vector_field_2d(points2d)
    u, v = points2d.T
    vect_u, vect_v = vects2d.T

    # Compute Jacobian
    r_squared = u**2 + v**2
    denom = 1 + r_squared
    denom_squared = denom**2

    # For x = 2u / (1 + u² + v²):
    dx_du = 2 * (1 + v**2 - u**2) / denom_squared
    dx_dv = -4 * u * v / denom_squared

    # For y = 2v / (1 + u² + v²):
    dy_du = -4 * u * v / denom_squared
    dy_dv = 2 * (1 + u**2 - v**2) / denom_squared

    # For z = (u² + v² - 1) / (1 + u² + v²):
    dz_du = 4 * u / denom_squared
    dz_dv = 4 * v / denom_squared

    # Apply the Jacobian: [v_x, v_y, v_z]^T = J × [v_u, v_v]^T
    vect_x = dx_du * vect_u + dx_dv * vect_v
    vect_y = dy_du * vect_u + dy_dv * vect_v
    vect_z = dz_du * vect_u + dz_dv * vect_v

    return np.array([vect_x, vect_y, vect_z]).T


def rotation_field(points3d, axis=IN):
    return np.cross(points3d, axis)


def flatten_field(points3d, vector_field_3d):
    vects = vector_field_3d(points3d)
    norms = normalize_along_axis(points3d, 1)
    return np.cross(vects, norms)


def get_sphereical_vector_field(
    v_func,
    axes,
    points,
    color=BLUE,
    stroke_width=1,
    mvltss=1.0,
    tip_width_ratio=4,
    tip_len_to_width=0.01,
):

    field = VectorField(
        v_func, axes,
        sample_coords=1.01 * points,
        max_vect_len_to_step_size=mvltss,
        density=1,
        stroke_width=stroke_width,
        tip_width_ratio=tip_width_ratio,
        tip_len_to_width=tip_len_to_width,
    )
    field.apply_depth_test()
    field.set_stroke(color, opacity=0.8)
    field.set_scale_stroke_with_zoom(True)
    return field


class SphereStreamLines(StreamLines):
    def __init__(self, func, coordinate_system, density=50, sample_coords=None, **kwargs):
        self.sample_coords = sample_coords
        super().__init__(func, coordinate_system, density=density, **kwargs)

    def get_sample_coords(self):
        if self.sample_coords is None:
            coords = fibonacci_sphere(int(4 * PI * self.density))
        else:
            coords = self.sample_coords
        return coords

    def draw_lines(self):
        super().draw_lines()
        for submob in self.submobjects:
            submob.set_points(normalize_along_axis(submob.get_points(), 1))
        return self


# Scenes

class IntroduceVectorField(InteractiveScene):
    def construct(self):
        # Set up sphere
        frame = self.frame
        self.camera.light_source.move_to([0, -10, 10])
        radius = 3
        axes = ThreeDAxes((-2, 2), (-2, 2), (-2, 2))
        axes.scale(radius)
        sphere = Sphere(radius=radius)
        sphere.set_color(BLUE_B, 0.3)
        sphere.always_sort_to_camera(self.camera)
        mesh = SurfaceMesh(sphere, (51, 25))
        mesh.set_stroke(WHITE, 1, 0.15)

        frame.reorient(0, 90, 0)
        self.play(
            frame.animate.reorient(30, 65, 0),
            ShowCreation(sphere),
            Write(mesh, time_span=(1, 3), lag_ratio=1e-2),
            run_time=3
        )

        # Tangent plane
        v_tracker = Point()
        v_tracker.move_to(radius * OUT)

        def place_on_vect(mobject):
            matrix = z_to_vector(v_tracker.get_center())
            mobject.set_points(np.dot(mobject.points_at_zenith, matrix.T))

        v_dot = Dot(radius=0.05)
        v_dot.set_fill(YELLOW)

        plane = Square(side_length=2 * radius)
        plane.set_stroke(WHITE, 1)
        plane.set_fill(GREY, 0.5)

        for mob in [plane, v_dot]:
            mob.move_to(radius * OUT)
            mob.points_at_zenith = mob.get_points().copy()
            mob.add_updater(place_on_vect)

        self.play(
            VFadeIn(v_dot),
            v_tracker.animate.move_to(radius * normalize(RIGHT + OUT)),
            run_time=2
        )
        self.wait()
        plane.update()
        plane.suspend_updating()
        self.play(
            FadeIn(plane),
            frame.animate.reorient(13, 78, 0),
            run_time=2
        )
        self.play(frame.animate.reorient(53, 60, 0, (-0.29, 0.09, 0.31), 8.97), run_time=2)
        self.wait()

        # Show one vector
        def v_func(points3d):
            v1 = stereographic_vector_field(points3d, right_func)
            v2 = normalize_along_axis(rotation_field(points3d, RIGHT), 1)
            v3 = normalize_along_axis(rotation_field(points3d, RIGHT + IN), 1)
            return (3 * v1 + v2 + v3) / 6

        def vector_field_3d(points3d):
            x, y, z = points3d.T
            return np.array([
                np.cos(3 * x) * np.sin(y),
                np.cos(5 * z) * np.sin(x),
                -z**2 + x,
            ]).T

        def alt_v_func(points3d):
            return normalize_along_axis(flatten_field(points3d, vector_field_3d), 1)

        def get_tangent_vect():
            origin = v_tracker.get_center()
            vector = Arrow(origin, origin + radius * v_func(normalize(origin).reshape(1, -1)).flatten(), buff=0, thickness=3)
            vector.set_fill(BLUE)
            vector.set_perpendicular_to_camera(frame)
            return vector

        tangent_vect = get_tangent_vect()
        self.add(plane, tangent_vect, v_dot)
        self.play(GrowArrow(tangent_vect))
        self.play(Rotate(tangent_vect, TAU, axis=v_tracker.get_center(), about_point=tangent_vect.get_start(), run_time=2))
        self.wait()

        # Show more vectors
        plane.resume_updating()
        tangent_vect.add_updater(lambda m: m.match_points(get_tangent_vect()))
        og_vect = rotate_vector(v_tracker.get_center(), 10 * DEG, axis=DOWN)
        v_tracker.clear_updaters()
        v_tracker.add_updater(lambda m, dt: m.rotate(60 * DEG * dt, axis=og_vect, about_point=ORIGIN))
        v_tracker.add_updater(lambda m, dt: m.rotate(1 * DEG * dt, axis=RIGHT, about_point=ORIGIN))

        frame.clear_updaters()
        frame.add_ambient_rotation(-2.5 * DEG)

        self.wait(2)

        field = self.get_vector_field(4000, axes, v_func, start_point=v_tracker.get_center())
        self.add(field, plane, v_tracker, tangent_vect, v_dot)
        self.play(
            ShowCreation(field),
            run_time=5
        )
        self.wait(4)
        self.play(
            FadeOut(plane),
            FadeOut(tangent_vect),
            FadeOut(v_dot),
            frame.animate.reorient(0, 59, 0, (-0.03, 0.15, -0.08), 6.80),
            run_time=4
        )
        self.wait(5)

        # Show denser field
        dots, dense_dots = [
            DotCloud(fibonacci_sphere(num), radius=0.01)
            for num in [4000, 400_000]
        ]
        for mob in [dots, dense_dots]:
            mob.make_3d()
            mob.set_color(WHITE)
            mob.scale(radius * 1.01)

        dense_field = self.get_vector_field(50_000, axes, v_func, mvltss=5.0)
        dense_field.set_stroke(opacity=0.35)

        dots.set_radius(0.02)
        dense_dots.set_radius(0.01)

        self.play(ShowCreation(dots, run_time=3))
        self.wait()
        self.play(
            FadeOut(dots, time_span=(1.5, 2.5)),
            FadeOut(field, time_span=(2.5, 3)),
            ShowCreation(dense_field),
            run_time=3
        )
        frame.clear_updaters()
        self.play(frame.animate.reorient(-66, 52, 0, (-0.51, 0.27, 0.2), 2.81), run_time=4)
        self.wait()

        # Show plane and tangent vector again
        field.save_state()
        field.set_stroke(width=1e-6)
        self.play(
            dense_field.animate.set_stroke(width=1e-6),
            Restore(field, time_span=(1, 3)),
            frame.animate.reorient(48, 54, 0, (-2.74, -2.49, -0.1), 11.93),
            VFadeIn(plane),
            VFadeIn(tangent_vect),
            run_time=5
        )
        self.wait(5)

        v_tracker.clear_updaters()
        null_point = 3 * normalize(np.array([-1, -0.25, 0.8]))
        self.play(
            v_tracker.animate.move_to(null_point),
            frame.animate.reorient(-43, 55, 0, (-2.42, 2.76, -0.55), 9.22),
            run_time=3
        )
        self.remove(tangent_vect)
        v_dot.update()
        self.play(
            FadeIn(v_dot, scale=0.25),
            FadeOut(plane, scale=0.25),
        )
        self.wait()

        # Define streamlines
        static_stream_lines = SphereStreamLines(
            lambda p: v_func(np.array(p).reshape(-1, 3)).flatten(),
            axes,
            density=400,
            stroke_width=2,
            magnitude_range=(0, 10),
            solution_time=1,
        )
        static_stream_lines.scale(radius)
        static_stream_lines.set_stroke(BLUE_B, 3, 0.8)
        stream_lines = AnimatedStreamLines(static_stream_lines, lag_range=10, rate_multiple=0.5)
        stream_lines.apply_depth_test()

        # Show the earth
        earth = TexturedSurface(sphere, "EarthTextureMap", "NightEarthTextureMap")
        earth.rotate(-90 * DEG, IN)
        earth.scale(1.001)
        frame.add_ambient_rotation(2 * DEG)

        self.add(sphere, earth, stream_lines, mesh, field)
        self.play(
            FadeOut(v_dot),
            FadeIn(earth, time_span=(0, 3)),
            field.animate.set_stroke(opacity=0.25).set_anim_args(time_span=(0, 2)),
            frame.aniamte.reorient(-90, 74, 0, (-1.37, 0.04, 0.37), 5.68),
        )
        self.wait(10)

        # Zoom in to null point
        frame.clear_updaters()
        dense_field = self.get_vector_field(10_000, axes, v_func)
        dense_field.set_stroke(opacity=0.35)
        self.play(
            FadeOut(field, run_time=2),
            FadeIn(dense_field, run_time=2),
            frame.animate.reorient(-69, 54, 0, (-0.01, 0.19, -0.04), 4.81),
            run_time=5
        )
        self.wait(3)
        self.play(
            frame.animate.reorient(129, 70, 0, (-0.18, 0.18, -0.1), 8.22),
            run_time=5
        )
        self.wait(10)

    def get_vector_field(self, axes, v_func, n_points, start_point=None, mvltss=1.0, random_order=False):
        points = fibonacci_sphere(n_points)

        if start_point is not None:
            points = points[np.argsort(np.linalg.norm(points - start_point.reshape(-1, 3), axis=1))]
        if random_order:
            indices = list(range(len(points)))
            random.shuffle(indices)
            points = points[indices]

        alpha = clip(inverse_interpolate(10_000, 1000, n_points), 0, 1)
        stroke_width = interpolate(1, 3, alpha**2)

        return get_sphereical_vector_field(v_func, axes, points, mvltss=mvltss, stroke_width=stroke_width)

    def old(self):
        # Vary the density
        density_tracker = ValueTracker(2000)
        field.add_updater(
            lambda m: m.become(self.get_vector_field(int(density_tracker.get_value()), axes, v_func))
        )
        self.add(field)
        field.resume_updating()
        frame.suspend_updating()
        self.play(
            density_tracker.animate.set_value(50_000),
            frame.animate.reorient(-30, 50, 0, (-0.07, 0.02, 0.36), 3.01),
            run_time=5,
        )
        field.suspend_updating()
        self.wait(3)


class ShowEastwardDirection(InteractiveScene):
    def construct(self):
        pass


class StereographicProjection(InteractiveScene):
    def construct(self):
        # Set up
        frame = self.frame
        x_max = 20
        axes = ThreeDAxes((-x_max, x_max), (-x_max, x_max), (-2, 2))
        plane = NumberPlane((-x_max, x_max), (-x_max, x_max))
        plane.background_lines.set_stroke(BLUE, 1, 0.5)
        plane.faded_lines.set_stroke(BLUE, 0.5, 0.25)
        axes.apply_depth_test()
        plane.apply_depth_test()

        sphere = Sphere(radius=1)
        sphere.set_opacity(0.5)
        sphere.always_sort_to_camera(self.camera)
        mesh = SurfaceMesh(sphere)
        mesh.set_stroke(WHITE, 1, 0.25)

        self.add(sphere, mesh, axes, plane)
        frame.reorient(-15, 64, 0, (0.0, 0.1, -0.09), 4.0)

        # Show the 2d cross section
        frame.clear_updaters()
        sphere.set_clip_plane(UP, 1)
        n_dots = 20
        sample_points = np.array([
            math.cos(theta) * OUT + math.sin(theta) * RIGHT
            for theta in np.linspace(0, TAU, n_dots + 2)[1:-1]
        ])
        sphere_dots, plane_dots, proj_lines = self.get_dots_and_lines(sample_points)

        self.play(
            sphere.animate.set_clip_plane(UP, 0),
            frame.animate.reorient(-43, 74, 0, (0.0, 0.0, -0.0), 3.50),
            FadeIn(sphere_dots, time_span=(1, 2)),
            ShowCreation(proj_lines, lag_ratio=0, time_span=(1, 2)),
            run_time=2
        )
        frame.add_ambient_rotation(2 * DEG)

        sphere_dot_ghosts = sphere_dots.copy().set_opacity(0.5)
        self.remove(sphere_dots)
        self.add(sphere_dot_ghosts)
        self.play(
            TransformFromCopy(sphere_dots, plane_dots, lag_ratio=0.5, run_time=10),
        )
        self.wait(3)

        planar_group = Group(sphere_dot_ghosts, plane_dots, proj_lines)

        # Show more points on the sphere
        sample_points = fibonacci_sphere(200)
        sphere_dots, plane_dots, proj_lines = self.get_dots_and_lines(sample_points)

        self.play(
            sphere.animate.set_clip_plane(UP, -1),
            frame.animate.reorient(-65, 73, 0, (-0.09, -0.01, -0.15), 5.08),
            ShowCreation(proj_lines, lag_ratio=0),
            FadeOut(planar_group),
            run_time=2,
        )
        self.wait(4)
        self.play(FadeIn(sphere_dots))
        self.wait(2)

        sphere_dot_ghosts = sphere_dots.copy().set_opacity(0.5)
        self.remove(sphere_dots)
        self.add(sphere_dot_ghosts)

        self.play(
            TransformFromCopy(sphere_dots, plane_dots, run_time=3),
        )
        self.wait(3)

        # Inverse projection
        plane.insert_n_curves(100)
        plane.save_state()
        proj_plane = plane.copy()
        proj_plane.apply_points_function(lambda p: inv_streographic_proj(p[:, :2]))
        proj_plane.make_smooth()
        proj_plane.background_lines.set_stroke(BLUE, 2, 1)
        proj_plane.faded_lines.set_stroke(BLUE, 1, 0.5)

        self.play(
            Transform(plane_dots, sphere_dot_ghosts),
            FadeOut(sphere_dot_ghosts, scale=0.9),
            Transform(plane, proj_plane),
            proj_lines.animate.set_stroke(opacity=0.2),
            run_time=4,
        )
        self.play(
            frame.animate.reorient(-20, 38, 0, (-0.04, -0.03, 0.13), 3.54),
            run_time=5
        )
        self.wait(5)
        self.play(
            frame.animate.reorient(-27, 73, 0, (-0.03, 0.03, 0.04), 5.27),
            Restore(plane),
            FadeOut(plane_dots),
            run_time=5
        )
        self.wait(2)

        # Show a vector field
        xy_field = VectorField(lambda ps: np.array([RIGHT for p in ps]), plane)
        xy_field.set_stroke(BLUE)
        xy_field.save_state()
        xy_field.set_stroke(width=1e-6)

        self.play(Restore(xy_field))
        self.wait(5)

        # Project the vector field up
        proj_field = xy_field.copy()
        proj_field.apply_points_function(lambda p: inv_streographic_proj(p[:, :2]), about_point=ORIGIN)
        proj_field.replace(sphere)
        proj_plane.background_lines.set_stroke(BLUE, 1, 0.5)
        proj_plane.faded_lines.set_stroke(BLUE, 0.5, 0.25)
        proj_plane.axes.set_stroke(WHITE, 0)

        self.play(
            Transform(plane, proj_plane),
            Transform(xy_field, proj_field),
            run_time=5,
        )
        self.play(
            frame.animate.reorient(-35, 31, 0, (0.05, 0.22, 0.22), 1.59),
            FadeOut(proj_lines),
            run_time=10
        )
        self.wait(8)

        # Show the flow (Maybe edit as a simple split-screen)
        proto_stream_lines = VGroup(
            Line([x, y, 0], [x + 20, y, 0]).insert_n_curves(25)
            for x in range(-100, 100, 10)
            for y in np.arange(-100, 100, 0.25)
        )
        for line in proto_stream_lines:
            line.virtual_time = 1
        proto_stream_lines.set_stroke(WHITE, 2, 0.8)
        proto_stream_lines.apply_points_function(lambda p: inv_streographic_proj(p[:, :2]), about_point=ORIGIN)
        proto_stream_lines.scale(1.01)
        proto_stream_lines.make_smooth()
        animated_lines = AnimatedStreamLines(proto_stream_lines, rate_multiple=0.2)

        sphere.set_color(GREY_E, 1)
        sphere.set_clip_plane(UP, 1)
        sphere.set_height(1.98).center()
        xy_field.apply_depth_test()
        animated_lines.apply_depth_test()
        self.add(sphere, mesh, plane, animated_lines, xy_field)
        self.play(
            FadeIn(sphere),
            FadeOut(xy_field),
            plane.animate.fade(0.25),
            xy_field.animate.set_stroke(opacity=0.5),
            frame.animate.reorient(-30, 29, 0, ORIGIN, 3.0),
            run_time=3
        )
        self.wait(30)
        return

    def get_dots_and_lines(self, sample_points, color=YELLOW, radius=0.025, stroke_opacity=0.35):
        sphere_dots = Group(TrueDot(point) for point in sample_points)
        for dot in sphere_dots:
            dot.make_3d()
            dot.set_color(color)
            dot.set_radius(radius)

        plane_dots = sphere_dots.copy().apply_points_function(stereographic_proj)
        proj_lines = VGroup(
            VGroup(
                Line(OUT, dot.get_center())
                for dot in dots
            )
            for dots in [plane_dots, sphere_dots]
        )
        proj_lines.set_stroke(color, 1, stroke_opacity)

        return sphere_dots, plane_dots, proj_lines

    def flow_with_projection_insertion(self):
        # For an insertion
        frame.clear_updaters()
        frame.reorient(-18, 77, 0, (-0.04, 0.04, 0.09), 5.43)
        frame.add_ambient_rotation(1 * DEG)
        sphere.set_clip_plane(UP, 2)
        sphere.set_color(GREY_D, 0.5)
        xy_field.apply_depth_test()
        xy_field.set_stroke(opacity=0.)
        proj_lines.set_stroke(opacity=0.35)
        self.add(xy_field, proj_lines, sphere, mesh)

        # Stream lines
        proto_stream_lines = VGroup(
            Line([x, y, 0], [x + 2, y, 0]).insert_n_curves(25)
            for x in np.arange(-x_max, x_max, 1)
            for y in np.arange(-x_max, x_max, 0.5)
        )
        for line in proto_stream_lines:
            line.virtual_time = 1
        proto_stream_lines.set_stroke(WHITE, 2, 0.8)

        sphere_stream_lines = proto_stream_lines.copy()
        sphere_stream_lines.apply_points_function(lambda p: inv_streographic_proj(p[:, :2]), about_point=ORIGIN)
        sphere_stream_lines.scale(1.01)
        sphere_stream_lines.make_smooth()

        animated_plane_lines = AnimatedStreamLines(proto_stream_lines, rate_multiple=0.2)
        animated_sphere_lines = AnimatedStreamLines(sphere_stream_lines, rate_multiple=0.2)

        self.add(animated_plane_lines)
        self.wait(3)
        for n in range(3 * 30):
            animated_sphere_lines.update(1 / 30)
        self.play(TransformFromCopy(animated_plane_lines, animated_sphere_lines), run_time=3)
        self.wait(5)

    def old(self):
        earth = TexturedSurface(sphere, "EarthTextureMap")
        earth.set_opacity(1)

        earth_group = Group(earth)
        earth_group.save_state()
        proj_earth = earth_group.copy()
        proj_earth.apply_points_function(stereographic_proj)
        proj_earth.interpolate(proj_earth, earth_group, 0.01)

        self.remove(earth_group)
        self.play(TransformFromCopy(earth_group, proj_earth), run_time=3)
        self.wait()
        self.remove(proj_earth)
        self.play(TransformFromCopy(proj_earth, earth_group), run_time=3)


class SimpleRightwardFlow(InteractiveScene):
    def construct(self):
        # Set up
        frame = self.frame
        x_max = 20
        axes = ThreeDAxes((-x_max, x_max), (-x_max, x_max), (-2, 2))
        plane = NumberPlane((-x_max, x_max), (-x_max, x_max))
        plane.background_lines.set_stroke(BLUE, 1, 0.5)
        plane.faded_lines.set_stroke(BLUE, 0.5, 0.25)
        axes.apply_depth_test()
        plane.apply_depth_test()

        # Simple flow
        frame.set_height(4)

        xy_field = VectorField(lambda ps: np.array([RIGHT for p in ps]), plane)
        xy_field.set_stroke(BLUE)
        self.add(xy_field)

        proto_stream_lines = VGroup(
            Line([x, y, 0], [x + 1, y, 0]).insert_n_curves(20)
            for x in np.arange(-10, 10, 0.5)
            for y in np.arange(-10, 10, 0.1)
        )
        for line in proto_stream_lines:
            line.virtual_time = 1
        proto_stream_lines.set_stroke(WHITE, 2, 0.8)

        animated_plane_lines = AnimatedStreamLines(proto_stream_lines, rate_multiple=0.2)

        self.add(animated_plane_lines)
        self.wait(30)


class SingleNullPointField(InteractiveScene):
    def construct(self):
        pass


class InsideOut(InteractiveScene):
    def construct(self):
        # Show sphere
        frame = self.frame
        self.camera.light_source.move_to([-3, 3, 3])
        radius = 3
        inner_scale = 0.999
        axes = ThreeDAxes((-5, 5), (-5, 5), (-5, 5))
        axes.set_stroke(WHITE, 1, 0.5)
        axes.apply_depth_test()
        axes.z_axis.rotate(0.1 * DEG, RIGHT)

        sphere = self.get_colored_sphere(radius, inner_scale)
        sphere.set_clip_plane(UP, radius)

        mesh = SurfaceMesh(sphere[0], resolution=(61, 31))
        mesh.set_stroke(WHITE, 1, 0.25)

        frame.reorient(-68, 70, 0)
        self.add(axes, sphere, mesh)
        self.play(sphere.animate.set_clip_plane(UP, 0), run_time=2)

        # Show point go to antipoode
        point = radius * normalize(LEFT + OUT)
        p_dot = TrueDot(point, color=YELLOW, radius=0.05).make_3d()
        p_label = Tex(R"p")
        p_label.rotate(90 * DEG, RIGHT).rotate(90 * DEG, IN)
        p_label.next_to(p_dot, OUT, SMALL_BUFF)

        neg_p_dot = p_dot.copy().move_to(-point)
        neg_p_label = Tex(R"-p")
        neg_p_label.rotate(90 * DEG, RIGHT)
        neg_p_label.next_to(neg_p_dot, IN, SMALL_BUFF)

        neg_p_dot.move_to(p_dot)

        dashed_line = DashedLine(point, -point, buff=0)
        dashed_line.set_stroke(YELLOW, 2)

        semi_circle = Arc(135 * DEG, 180 * DEG, radius=radius)
        semi_circle.set_stroke(YELLOW, 3)
        semi_circle.rotate(90 * DEG, RIGHT, about_point=ORIGIN)
        dashed_semi_circle = DashedVMobject(semi_circle, num_dashes=len(dashed_line))

        self.play(
            FadeIn(p_dot, scale=0.5),
            Write(p_label),
        )
        self.play(
            ShowCreation(dashed_semi_circle),
            MoveAlongPath(neg_p_dot, semi_circle, rate_func=linear),
            TransformFromCopy(p_label, neg_p_label, time_span=(2, 3)),
            Rotate(p_label, 90 * DEG, OUT),
            frame.animate.reorient(59, 87, 0),
            run_time=3,
        )
        frame.add_ambient_rotation(-2 * DEG)
        self.wait(2)
        self.play(ReplacementTransform(dashed_semi_circle, dashed_line))
        self.wait(3)

        # Show more antipodes
        angles = np.linspace(0, 90 * DEG, 15)[1:]
        top_dots, low_dots, lines = groups = [
            Group(
                template.copy().rotate(angle, axis=UP, about_point=ORIGIN)
                for angle in angles
            )
            for template in [p_dot, neg_p_dot, dashed_line]
        ]
        for group in groups:
            group.set_submobject_colors_by_gradient(YELLOW, BLUE, interp_by_hsl=True)

        self.play(FadeIn(top_dots, lag_ratio=0.1))
        frame.clear_updaters()
        self.play(
            LaggedStartMap(ShowCreation, lines, lag_ratio=0.25),
            LaggedStart(
                (TransformFromCopy(top_dot, low_dot, rate_func=linear)
                for top_dot, low_dot in zip(top_dots, low_dots)),
                lag_ratio=0.25,
                group_type=Group,
            ),
            frame.animate.reorient(-49, 60, 0),
            run_time=7
        )

        # Just show the cap
        cap = self.get_colored_sphere(radius, inner_scale, v_range=(0.75 * PI, PI))
        frame.clear_updaters()

        self.add(cap, sphere, mesh)
        self.play(
            FadeOut(p_label),
            FadeOut(neg_p_label),
            FadeOut(top_dots),
            FadeOut(lines),
            FadeOut(low_dots),
            FadeOut(p_dot),
            FadeOut(neg_p_dot),
            FadeOut(dashed_line),
            FadeOut(sphere, 0.1 * UP),
            FadeIn(cap),
        )
        self.wait()
        self.play(frame.animate.reorient(-48, 93, 0), run_time=3)
        self.wait()

        # Show transition to antipode
        anti_cap = cap.copy().rotate(PI, axis=OUT, about_point=ORIGIN).stretch(-1, 2, about_point=ORIGIN)
        anti_cap[0].shift(1e-2 * OUT)
        all_points = cap[0].get_points()
        indices = random.sample(list(range(len(all_points))), 200)
        pre_points = all_points[indices]
        post_points = -1 * pre_points

        antipode_lines = VGroup(
            Line(point, -point)
            for point in pre_points
        )
        antipode_lines.set_stroke(YELLOW, 1, 0.25)
        antipode_lines.apply_depth_test()

        def update_lines(lines):
            points1 = cap[0].get_points()[indices]
            points2 = anti_cap[0].get_points()[indices]
            for line, p1, p2 in zip(lines, points1, points2):
                line.put_start_and_end_on(p1, p2)

        antipode_lines.add_updater(update_lines)

        rot_arcs = VGroup(
            Arrow(4 * RIGHT, 4 * LEFT, path_arc=180 * DEG, thickness=5),
            Arrow(4 * LEFT, 4 * RIGHT, path_arc=180 * DEG, thickness=5),
        )
        flip_arrows = VGroup(
            Arrow(3 * IN, 3.2 * OUT, thickness=5),
            Arrow(3 * OUT, 3.2 * IN, thickness=5),
        ).rotate(90 * DEG).shift(4 * RIGHT)

        self.play(
            ShowCreation(antipode_lines, lag_ratio=0, suspend_mobject_updating=True),
            frame.animate.reorient(8, 79, 0, (0.0, 0.02, 0.0)),
            run_time=4,
        )
        self.wait()
        self.play(
            Write(rot_arcs, lag_ratio=0, run_time=1),
            Rotate(cap, PI, axis=OUT, run_time=3, about_point=ORIGIN),
            Rotate(mesh, PI, axis=OUT, run_time=3, about_point=ORIGIN),
        )
        self.play(FadeOut(rot_arcs))
        self.play(
            FadeIn(flip_arrows, time_span=(0, 1)),
            Transform(cap, anti_cap),
            mesh.animate.stretch(-1, 2, about_point=ORIGIN),
            VFadeOut(antipode_lines),
            run_time=3
        )
        self.play(FadeOut(flip_arrows))
        self.play(frame.animate.reorient(-9, 96, 0, (0.0, 0.02, 0.0)), run_time=5)
        self.wait()

        # Antipode homotopy

    def get_colored_sphere(
        self,
        radius=3,
        inner_scale=0.999,
        outer_color=BLUE_E,
        inner_color=GREY_BROWN,
        u_range=(0, TAU),
        v_range=(0, PI),
    ):
        outer_sphere = Sphere(radius=radius, u_range=u_range, v_range=v_range)
        inner_sphere = outer_sphere.copy()
        outer_sphere.set_color(outer_color, 1)
        inner_sphere.set_color(inner_color, 1)
        inner_sphere.scale(inner_scale)
        return Group(outer_sphere, inner_sphere)

    def old_homotopy(self):
        def homotopy(x, y, z, t, scale=-1):
            p = np.array([x, y, z])
            power = 1 + 0.2 * (x / radius)
            return interpolate(p, scale * p, t**power)

        antipode_lines = VGroup(
            Line(point, -point)
            for point in random.sample(list(cap[0].get_points()), 100)
        )
        antipode_lines.set_stroke(YELLOW, 1, 0.25)

        self.play(
            Homotopy(lambda x, y, z, t: homotopy(x, y, z, t, -inner_scale), cap[0]),
            Homotopy(lambda x, y, z, t: homotopy(x, y, z, t, -1.0 / inner_scale), cap[1]),
            ShowCreation(antipode_lines, lag_ratio=0),
            frame.animate.reorient(-65, 68, 0),
            run_time=3
        )
        self.play(FadeOut(antipode_lines))
        self.play(frame.animate.reorient(-48, 148, 0), run_time=3)
        self.wait()
        self.play(frame.animate.reorient(-70, 83, 0), run_time=3)
        self.play(frame.animate.reorient(-124, 77, 0), run_time=10)


class DefineOrientation(InsideOut):
    def construct(self):
        # Latitude and Longetude
        frame = self.frame
        radius = 3
        sphere = Sphere(radius=radius)
        sphere.set_color(GREY_E, 1)
        earth = TexturedSurface(sphere, "EarthTextureMap", "NightEarthTextureMap")
        earth.set_opacity(0.5)
        mesh = SurfaceMesh(sphere, resolution=(61, 31))
        mesh.set_stroke(WHITE, 1, 0.25)

        uv_tracker = ValueTracker(np.array([180 * DEG, 90 * DEG]))
        dot = TrueDot()
        dot.set_color(YELLOW)
        dot.add_updater(lambda m: m.move_to(sphere.uv_func(*uv_tracker.get_value())))
        dot.set_z_index(2)

        lat_label, lon_label = labels = VGroup(
            Tex(R"\text{Lat: }\, 10^\circ"),
            Tex(R"\text{Lon: }\, 10^\circ"),
        )
        labels.arrange(DOWN, aligned_edge=LEFT)
        labels.fix_in_frame()
        labels.to_corner(UL)
        lat_label.make_number_changeable("10", edge_to_fix=RIGHT).add_updater(
            lambda m: m.set_value(np.round((uv_tracker.get_value()[1] - 90 * DEG) / DEG))
        )
        lon_label.make_number_changeable("10", edge_to_fix=RIGHT).add_updater(
            lambda m: m.set_value(np.round((uv_tracker.get_value()[0] - 180 * DEG) / DEG))
        )
        labels.add_updater(lambda m: m.fix_in_frame())

        self.add(sphere, mesh)
        self.add(labels)
        frame.reorient(-66, 85, 0, (-0.06, 0.18, 0.06), 6.78)
        self.play(FadeIn(dot))

        lon_line = TracedPath(dot.get_center, stroke_color=RED)
        self.add(lon_line, dot)
        self.play(uv_tracker.animate.increment_value([0, 30 * DEG]), run_time=4)
        lon_line.suspend_updating()

        lat_line = TracedPath(dot.get_center, stroke_color=TEAL)
        self.add(lat_line, dot)
        self.play(uv_tracker.animate.increment_value([42 * DEG, 0]), run_time=4)
        lat_line.suspend_updating()

        # Show tangent vectors
        u, v = uv_tracker.get_value()
        epsilon = 1e-4
        point = sphere.uv_func(u, v)
        u_step = normalize(sphere.uv_func(u + epsilon, v) - point)
        v_step = normalize(sphere.uv_func(u, v + epsilon) - point)

        u_vect = Arrow(point, point + 0.5 * u_step, buff=0, thickness=2).set_color(TEAL)
        v_vect = Arrow(point, point + 0.5 * v_step, buff=0, thickness=2).set_color(RED)
        tangent_vects = VGroup(u_vect, v_vect)
        tangent_vects.set_z_index(1)
        tangent_vects.set_fill(opacity=0.8)
        for vect in tangent_vects:
            vect.always.set_perpendicular_to_camera(frame)

        self.play(
            frame.animate.reorient(-47, 60, 0, (-0.51, 0.42, 0.14), 4.33),
            dot.animate.scale(0.5),
            LaggedStartMap(GrowArrow, tangent_vects, lag_ratio=0.5),
            run_time=3
        )
        tangent_vects.clear_updaters()

        lat_line2 = TracedPath(dot.get_center, stroke_color=TEAL)
        self.add(lat_line2)
        self.play(
            uv_tracker.animate.increment_value([30 * DEG, 0]).set_anim_args(rate_func=wiggle),
            FadeOut(lon_line),
            run_time=3
        )
        lat_line2.clear_updaters()
        self.wait()

        lon_line2 = TracedPath(dot.get_center, stroke_color=RED)
        self.add(lon_line2, tangent_vects)
        self.play(
            uv_tracker.animate.increment_value([0, 30 * DEG]).set_anim_args(rate_func=wiggle),
            FadeOut(lat_line),
            run_time=3
        )
        lon_line2.clear_updaters()

        # Show normal vector
        normal_vect = Arrow(
            point, point + 0.5 * np.cross(u_step, v_step),
            thickness=2,
            buff=0
        )
        normal_vect.set_fill(BLUE, 0.8)
        normal_vect.rotate(90 * DEG, axis=normal_vect.get_vector())

        self.play(
            GrowArrow(normal_vect, time_span=(0, 2)),
            FadeOut(labels, time_span=(0, 2)),
            frame.animate.reorient(-103, 63, 0, (-0.12, -0.36, 0.43), 4.75),
            run_time=8
        )

        # Show a full vector field
        normal_field = get_sphereical_vector_field(
            lambda p: p,
            ThreeDAxes(),
            points=np.array([
                sphere.uv_func(u, v)
                for u in np.arange(0, TAU, TAU / 60)
                for v in np.arange(0, PI, PI / 30)
            ])
        )

        normal_field.save_state()
        normal_field.set_stroke(width=1e-6)

        self.play(
            Restore(normal_field),
            frame.animate.reorient(-103, 62, 0, (-0.09, 0.25, 0.23), 7.26),
            run_time=3
        )
        self.wait()
        self.play(
            FadeOut(normal_field),
            FadeOut(sphere, scale=0.9),
            VGroup(lat_line2, lon_line2).animate.set_stroke(width=1)
        )

        # Show antipode map
        group = Group(lat_line2, lon_line2, tangent_vects, dot)
        anti_group = group.copy().scale(-1, min_scale_factor=-np.inf, about_point=ORIGIN)

        antipode_lines = VGroup(
            Line(p1, p2)
            for index in [0, 1]
            for p1, p2 in zip(group[index].get_points(), anti_group[index].get_points())
        )
        antipode_lines.set_stroke(YELLOW, 1, 0.1)

        self.play(
            TransformFromCopy(group, anti_group),
            frame.animate.reorient(-194, 105, 0, (0.7, -0.22, -0.61), 4.64),
            ShowCreation(antipode_lines, lag_ratio=0),
            run_time=5
        )
        self.play(antipode_lines.animate.set_stroke(opacity=0.02))
        self.wait()

        # New normal
        new_normal = normal_vect.copy()
        new_normal.shift(-2 * point)

        self.play(
            GrowArrow(new_normal),
            frame.animate.reorient(-197, 90, 0, (1.16, -0.3, -0.99), 4.64),
            run_time=3
        )
        self.wait()

        # Show reverserd vector field
        anti_normal_field = get_sphereical_vector_field(
            lambda p: -p,
            ThreeDAxes(),
            points=np.array([
                sphere.uv_func(u, v)
                for u in np.arange(0, TAU, TAU / 60)
                for v in np.arange(0, PI, PI / 30)
            ])
        )

        self.play(
            FadeIn(anti_normal_field, time_span=(0, 2)),
            frame.animate.reorient(-167, 76, 0, (0.13, -1.02, -0.2), 8.43),
            run_time=10
        )


class FlowingWater(InteractiveScene):
    def construct(self):
        # Set up axes
        radius = 2
        frame = self.frame
        axes = ThreeDAxes((-3, 3), (-3, 3), (-3, 3))
        axes.scale(radius)
        frame.reorient(-89, 77, 0)
        frame.add_ambient_rotation(2 * DEG)
        self.add(axes)

        # Add water
        water = DotCloud()
        water.set_radius(0.02)
        water.set_color(BLUE)
        water_opacity_tracker = ValueTracker(0.2)
        water_radius_tracker = ValueTracker(0.02)

        def add_random_points(water, n_points=10, sigma=1):
            new_points = np.random.normal(0, sigma, (n_points, 3))
            water.append_points(new_points)

        def flow_out(water, dt, velocity=10, n_refreshes=1000):
            if dt == 0:
                pass
            points = water.get_points()
            radii = np.linalg.norm(points, axis=1)
            denom = 4 * PI * radii**2
            denom[denom == 0] = 1
            vels = points / denom[:, np.newaxis]
            new_points = points + velocity * vels * dt

            indices = np.random.randint(0, len(points), n_refreshes)
            new_points[indices] = np.random.normal(0, 0.5, (n_refreshes, 3))
            water.set_points(new_points)
            water.set_opacity(water_opacity_tracker.get_value() / np.clip(radii, 1, np.inf))
            water.set_radius(water_radius_tracker.get_value())
            return water

        add_random_points(water, 500_000, sigma=3)
        water.add_updater(flow_out)

        source_dot = GlowDot(ORIGIN, color=BLUE)

        self.add(source_dot, water)
        self.wait(25)

        # Show full sphere
        sphere = Sphere(radius=radius)
        sphere.set_color(GREY, 0.25)
        sphere.always_sort_to_camera(self.camera)
        mesh = SurfaceMesh(sphere, resolution=(61, 31))
        mesh.set_stroke(WHITE, 1, 0.5)
        mesh.set_z_index(2)

        def get_unit_normal_field(u_range, v_range):
            return get_sphereical_vector_field(
                lambda p: p,
                ThreeDAxes(),
                points=np.array([
                    sphere.uv_func(u, v)
                    for u in u_range
                    for v in v_range
                ])
            )

        normal_field = get_unit_normal_field(
            np.arange(0, TAU, TAU / 60),
            np.arange(0, PI, PI / 30),
        )

        self.play(
            ShowCreation(sphere),
            Write(mesh, lag_ratio=1e-3),
        )
        self.play(FadeIn(normal_field))
        self.wait(5)

        # Show single patch
        u_range_params = (0 * DEG, 15 * DEG, 5 * DEG)
        v_range_params = (120 * DEG, 130 * DEG, 5 * DEG)
        patch = Sphere(
            radius=radius,
            u_range=u_range_params[:2],
            v_range=v_range_params[:2],
        )
        patch.set_color(WHITE, 0.6)

        patch_normals = get_unit_normal_field(np.arange(*u_range_params), np.arange(*v_range_params))

        self.play(
            FadeOut(sphere, time_span=(0, 1)),
            FadeOut(normal_field, time_span=(0, 1)),
            FadeIn(patch, time_span=(0, 1)),
            FadeIn(patch_normals, time_span=(0, 1)),
            mesh.animate.set_stroke(opacity=0.1),
            frame.animate.reorient(19, 55, 0, (1.65, 0.28, 0.98), 2.32),
            water_opacity_tracker.animate.set_value(0.1),
            water_radius_tracker.animate.set_value(0.01),
            run_time=5
        )
        frame.clear_updaters()
        self.wait(7.5)

        patch_group = Group(patch, patch_normals)
        self.play(
            Rotate(patch_group, PI, axis=UP),
            run_time=2
        )
        self.wait(1)
        self.play(
            Rotate(patch_group, PI, axis=UP, time_span=(0, 1)),
            FadeIn(sphere),
            FadeIn(normal_field),
            frame.animate.reorient(30, 78, 0, (0.19, -0.03, 0.09), 5.67),
            water_opacity_tracker.animate.set_value(0.15),
            water_radius_tracker.animate.set_value(0.015),
            run_time=5
        )
        self.play(FadeOut(patch_group))
        frame.add_ambient_rotation(-2 * DEG)
        self.wait(20)
        self.play(FadeOut(normal_field))

        # Show deformations
        sphere.always_sort_to_camera(self.camera)
        sphere_group = Group(sphere, mesh)

        def random_deformation(points, seed=0):
            random.seed(seed)
            x, y, z = points.T
            wiggle_size = 0.75
            max_freq = 5
            z += wiggle_size * random.random() * np.cos(max_freq * random.random() * x)
            x += wiggle_size * random.random() * np.cos(max_freq * random.random() * y)
            y += wiggle_size * random.random() * np.cos(max_freq * random.random() * z)

            return np.array([x, y, z]).T

        sphere_group_bases = [
            sphere_group.copy(),
            sphere_group.copy().apply_points_function(random_deformation),
            sphere_group.copy().apply_points_function(lambda p: random_deformation(p, seed=1)),
        ]

        sphere_group.time = 0

        center_tracker = Point()

        def update_sphere_group(sphere_group, dt):
            alpha1 = np.sin(0.5 * sphere_group.time)**2
            alpha2 = np.sin(0.55 * sphere_group.time)**2
            mob_tups = zip(
                sphere_group.family_members_with_points(),
                sphere_group_bases[0].family_members_with_points(),
                sphere_group_bases[1].family_members_with_points(),
                sphere_group_bases[2].family_members_with_points(),
            )
            for m0, m1, m2, m3 in mob_tups:
                ps = interpolate(m1.get_points(), m2.get_points(), alpha1)
                ps = interpolate(ps, m3.get_points(), alpha2)
                m0.set_points(ps)
            sphere_group.move_to(center_tracker)
            sphere_group.time += dt
            return sphere_group

        sphere_group.add_updater(update_sphere_group)

        self.add(sphere_group)
        self.wait(8)
        self.play(
            center_tracker.animate.shift(1.25 * radius * RIGHT),
            run_time=6,
            rate_func=there_and_back_with_pause,
        )
        self.wait(8)


class SurfaceTestForSenia(InteractiveScene):
    def construct(self):
        # Test
        axes = ThreeDAxes()

        time_tracker = ValueTracker(0)

        def get_surface(t):
            return ParametricSurface(
                lambda u, v: spherical_eversion(u, v, t),
                u_range=(0, 2 * PI),
                v_range=(0.1, PI),
                resolution=(25, 25),
            )

        def update_surface(surface):
            surface.match_points(get_surface(time_tracker.get_value()))

        surface = TexturedSurface(get_surface(0), "Tower2")

        self.frame.reorient(-3, 60, 0, (-0.02, 0.05, 0.09), 4.26)  # Use shift-D to copy frame state
        self.add(axes)
        self.add(surface)
        self.play(
            time_tracker.animate.set_value(1).set_anim_args(rate_func=linear),
            UpdateFromFunc(surface, update_surface),
            run_time=20,
        )
        self.wait()

        self.play(
            ShowCreation(surface),
            self.frame.animate.reorient(41, 70, 0, (-0.17, 0.22, 0.06), 5.33)
        )