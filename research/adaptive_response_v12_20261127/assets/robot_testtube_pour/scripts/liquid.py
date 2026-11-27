"""Fixed-particle GPU position-based liquid for the robot pouring example.

Load inside Isaac Sim (or a Python environment with Newton and Warp).
This module creates no SimulationApp, USD stage, callbacks, or emitters.
"""

_PBF_KERNEL_CACHE = None

def _build_pbf_kernels():
    """Compile lazily, after the host has made Warp/Newton available."""
    global _PBF_KERNEL_CACHE
    if _PBF_KERNEL_CACHE is not None:
        return _PBF_KERNEL_CACHE
    import warp as wp

    @wp.func
    def pbf_weight(r2: float, h2: float, coefficient: float):
        v = wp.max(h2 - r2, 0.0)
        return coefficient * v * v * v

    @wp.func
    def pbf_gradient(delta: wp.vec3, h: float, coefficient: float):
        r = wp.length(delta)
        result = wp.vec3(0.0)
        if r > 1.0e-8 and r < h:
            result = coefficient * (h - r) * (h - r) * delta / r
        return result

    @wp.func
    def tube_distance(point: wp.vec3, pose: wp.transform, dimensions: wp.vec4):
        p = wp.transform_point(wp.transform_inverse(pose), point)
        r, length, thickness = dimensions[0], dimensions[1], dimensions[2]
        radial = wp.length(wp.vec2(p[0], p[1]))
        radial_normal = wp.vec3(p[0], p[1], 0.0) / wp.max(radial, 1.0e-8)
        inner = radial - r
        outer = radial - (r + thickness)
        inner_normal = radial_normal
        outer_normal = radial_normal
        if p[2] < r:
            offset = p - wp.vec3(0.0, 0.0, r)
            distance = wp.length(offset)
            inner = distance - r
            outer = distance - (r + thickness)
            inner_normal = offset / wp.max(distance, 1.0e-8)
            outer_normal = inner_normal
        else:
            top = p[2] - length
            if top > outer:
                outer = top
                outer_normal = wp.vec3(0.0, 0.0, 1.0)
        distance = outer
        normal = outer_normal
        if -inner > outer:
            distance = -inner
            normal = -inner_normal
        return distance, wp.quat_rotate(wp.transform_get_rotation(pose), normal)

    @wp.func
    def beaker_distance(point: wp.vec3, center_radius: wp.vec4, height_thickness: wp.vec2):
        p = point - wp.vec3(center_radius[0], center_radius[1], center_radius[2])
        r, height, thickness = center_radius[3], height_thickness[0], height_thickness[1]
        radial = wp.length(wp.vec2(p[0], p[1]))
        radial_normal = wp.vec3(p[0], p[1], 0.0) / wp.max(radial, 1.0e-8)
        outer = radial - (r + thickness)
        outer_normal = radial_normal
        if p[2] - height > outer:
            outer = p[2] - height
            outer_normal = wp.vec3(0.0, 0.0, 1.0)
        if -p[2] - thickness > outer:
            outer = -p[2] - thickness
            outer_normal = wp.vec3(0.0, 0.0, -1.0)
        cavity = radial - r
        cavity_normal = radial_normal
        if -p[2] > cavity:
            cavity = -p[2]
            cavity_normal = wp.vec3(0.0, 0.0, -1.0)
        distance, normal = outer, outer_normal
        if -cavity > outer:
            distance, normal = -cavity, -cavity_normal
        return distance, normal

    @wp.func
    def solid_distance(point: wp.vec3, tube: wp.vec4, beaker: wp.vec4,
                       beaker_ext: wp.vec2, body_q: wp.array[wp.transform]):
        distance, normal = point[2], wp.vec3(0.0, 0.0, 1.0)
        d, n = tube_distance(point, body_q[0], tube)
        if d < distance:
            distance, normal = d, n
        d, n = beaker_distance(point, beaker, beaker_ext)
        if d < distance:
            distance, normal = d, n
        return distance, normal

    @wp.func
    def wall_support(point: wp.vec3, h: float, normalization: float, tube: wp.vec4,
                     beaker: wp.vec4, beaker_ext: wp.vec2, body_q: wp.array[wp.transform]):
        distance, normal = solid_distance(point, tube, beaker, beaker_ext, body_q)
        fraction = float(0.0)
        gradient = wp.vec3(0.0)
        if distance < h:
            s = wp.clamp(distance / h, 0.0, 1.0)
            s2 = s * s
            primitive = s * (1.0 + s2 * (-4.0 / 3.0 + s2 * (6.0 / 5.0 + s2 * (-4.0 / 7.0 + s2 / 9.0))))
            fraction = normalization * wp.max(0.0, 0.5 - (315.0 / 256.0) * primitive)
            v = 1.0 - s2
            gradient = normal * (-normalization * (315.0 / (256.0 * h)) * v * v * v * v)
        return fraction, gradient

    @wp.kernel
    def project_solids(position: wp.array[wp.vec3], tube: wp.vec4, beaker: wp.vec4,
                       beaker_ext: wp.vec2, body_q: wp.array[wp.transform], clearance: float):
        i = wp.tid()
        p = position[i]
        for iteration in range(3):
            distance, normal = solid_distance(p, tube, beaker, beaker_ext, body_q)
            if distance < clearance:
                p += (clearance - distance) * normal
        p[2] = wp.max(p[2], clearance)
        position[i] = p

    @wp.kernel
    def predict(old_position: wp.array[wp.vec3], old_velocity: wp.array[wp.vec3],
                position: wp.array[wp.vec3], velocity: wp.array[wp.vec3],
                dt: float, clearance: float):
        i = wp.tid()
        v = old_velocity[i] + wp.vec3(0.0, 0.0, -9.81) * dt
        p = old_position[i] + dt * v
        if p[2] < clearance:
            p[2] = clearance
            v[2] = wp.max(v[2], 0.0)
        position[i] = p
        velocity[i] = v

    @wp.kernel
    def density_lambdas(position: wp.array[wp.vec3], grid: wp.uint64,
                        h: float, volume: float, weight_coefficient: float,
                        gradient_coefficient: float, normalization: float,
                        tube: wp.vec4, beaker: wp.vec4, beaker_ext: wp.vec2,
                        body_q: wp.array[wp.transform],
                        lambdas: wp.array[float], density_ratio: wp.array[float]):
        i = wp.tid()
        p = position[i]
        density = float(0.0)
        gradient_self = wp.vec3(0.0)
        denominator = float(0.0)
        neighbors = wp.hash_grid_query(grid, p, h)
        j = int(0)
        while wp.hash_grid_query_next(neighbors, j):
            delta = p - position[j]
            density += volume * pbf_weight(wp.length_sq(delta), h * h, weight_coefficient)
            if i != j:
                gradient = volume * pbf_gradient(delta, h, gradient_coefficient)
                gradient_self += gradient
                denominator += wp.length_sq(gradient)
        fraction, boundary_gradient = wall_support(p, h, normalization, tube, beaker, beaker_ext, body_q)
        density += fraction
        gradient_self += boundary_gradient
        denominator += wp.length_sq(gradient_self)
        density_ratio[i] = density
        constraint = wp.max(density - 1.0, 0.0)
        lambdas[i] = -constraint / (denominator + 1.0e-6 / (h * h))

    @wp.kernel
    def pressure_delta(position: wp.array[wp.vec3], grid: wp.uint64,
                       h: float, volume: float, gradient_coefficient: float,
                       normalization: float, tube: wp.vec4, beaker: wp.vec4,
                       beaker_ext: wp.vec2, body_q: wp.array[wp.transform],
                       lambdas: wp.array[float], delta_position: wp.array[wp.vec3],
                       maximum_delta: float):
        i = wp.tid()
        p = position[i]
        correction = wp.vec3(0.0)
        neighbors = wp.hash_grid_query(grid, p, h)
        j = int(0)
        while wp.hash_grid_query_next(neighbors, j):
            if i != j:
                correction += volume * (lambdas[i] + lambdas[j]) * pbf_gradient(
                    p - position[j], h, gradient_coefficient
                )
        fraction, gradient = wall_support(p, h, normalization, tube, beaker, beaker_ext, body_q)
        correction += lambdas[i] * gradient
        length = wp.length(correction)
        if length > maximum_delta:
            correction *= maximum_delta / length
        delta_position[i] = correction

    @wp.kernel
    def apply_delta(position: wp.array[wp.vec3], delta_position: wp.array[wp.vec3],
                    clearance: float):
        i = wp.tid()
        p = position[i] + delta_position[i]
        # The tabletop is a geometric plane at z=0, not a container mask.
        p[2] = wp.max(p[2], clearance)
        position[i] = p

    @wp.kernel
    def update_velocity(old_position: wp.array[wp.vec3], position: wp.array[wp.vec3],
                        velocity: wp.array[wp.vec3], dt: float):
        i = wp.tid()
        velocity[i] = (position[i] - old_position[i]) / dt

    @wp.kernel
    def viscosity_velocity(position: wp.array[wp.vec3], velocity: wp.array[wp.vec3],
                           output: wp.array[wp.vec3], grid: wp.uint64, h: float,
                           volume: float, weight_coefficient: float, amount: float):
        i = wp.tid()
        p = position[i]
        change = wp.vec3(0.0)
        neighbors = wp.hash_grid_query(grid, p, h)
        j = int(0)
        while wp.hash_grid_query_next(neighbors, j):
            if i != j:
                weight = volume * pbf_weight(wp.length_sq(p - position[j]), h * h, weight_coefficient)
                change += weight * (velocity[j] - velocity[i])
        output[i] = velocity[i] + amount * change

    _PBF_KERNEL_CACHE = (predict, density_lambdas, pressure_delta, apply_delta,
                         update_velocity, viscosity_velocity, project_solids)
    return _PBF_KERNEL_CACHE


class PourFluid:
    """PBF liquid with native Newton mesh contacts and surface extraction.

    pose_fn(t) returns (position_xyz, quaternion_xyzw) for the tube's local frame.
    Tube/beaker mesh pairs are (vertices_Nx3, flattened_triangle_indices).
    labels retain each initial particle's group for conservation/transfer metrics.
    """

    def __init__(
        self, initial_points, labels, tube_mesh, beaker_mesh, pose_fn,
        *, spacing=0.0035, voxel=0.007, dt=1.0 / 60.0, substeps=4,
        density=1000.0, viscosity=0.001, max_iterations=32,
        surface_voxel=0.0045, surface_max_cells=1_000_000,
    ):
        import numpy as np
        import warp as wp
        import newton
        from newton.solvers import SolverImplicitMPM

        self.np, self.wp = np, wp
        self.pose_fn = pose_fn
        self.spacing, self.voxel = float(spacing), float(voxel)
        self.dt, self.substeps = float(dt), int(substeps)
        if not all(np.isfinite(v) and v > 0 for v in (spacing, voxel, dt, density, surface_voxel)):
            raise ValueError("Spacing, voxel sizes, dt, and density must be finite and positive.")
        if self.substeps < 1 or self.substeps != substeps:
            raise ValueError("substeps must be a positive integer.")
        if not np.isfinite(viscosity) or viscosity < 0:
            raise ValueError("viscosity must be finite and nonnegative.")
        if int(max_iterations) != max_iterations or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer.")
        points = np.asarray(initial_points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError("initial_points must be a finite nonempty Nx3 array.")
        self.labels = np.asarray(labels).copy()
        if self.labels.shape != (len(points),):
            raise ValueError("labels must have one entry per initial particle.")
        self.labels.setflags(write=False)
        self.particle_count = len(points)
        self.particle_mass = float(density * self.spacing ** 3)
        self.total_mass = self.particle_count * self.particle_mass
        self._frame_index = 0
        self.sub_dt = self.dt / self.substeps

        def checked_mesh(pair, name):
            vertices, indices = pair
            vertices = np.asarray(vertices, dtype=np.float32)
            indices = np.asarray(indices, dtype=np.int32).reshape(-1)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
                raise ValueError(f"{name} vertices must be finite Nx3.")
            if not len(indices) or len(indices) % 3 or indices.min() < 0 or indices.max() >= len(vertices):
                raise ValueError(f"{name} must have valid flattened triangle indices.")
            return newton.Mesh(vertices, indices, compute_inertia=False, is_solid=True)

        p0, q0 = self._sample_pose(0.0)
        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)
        self.tube_body = builder.add_body(
            xform=wp.transform(wp.vec3(*p0), wp.quat(*q0)),
            mass=0.0, is_kinematic=True, label="pour_tube",
        )
        collision = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0)
        builder.add_shape_mesh(
            body=self.tube_body, mesh=checked_mesh(tube_mesh, "tube"),
            cfg=collision, label="pour_tube_shell",
        )
        builder.add_shape_mesh(
            body=-1, mesh=checked_mesh(beaker_mesh, "beaker"),
            cfg=collision, label="pour_beaker_shell",
        )
        builder.add_shape_plane(width=2.0, length=2.0, cfg=collision)
        builder.add_particles(
            pos=points.tolist(), vel=np.zeros_like(points).tolist(),
            mass=[self.particle_mass] * self.particle_count,
            radius=[0.5 * self.spacing] * self.particle_count,
        )
        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))
        self.model.mpm.friction.fill_(0.0)
        self.model.mpm.viscosity.fill_(float(viscosity))
        # No yield shear stress or friction: a liquid, not a granular material.
        self.model.mpm.yield_stress.fill_(0.0)
        self.model.mpm.tensile_yield_ratio.fill_(0.1)

        config = SolverImplicitMPM.Config()
        config.voxel_size = self.voxel
        config.max_iterations = int(max_iterations)
        config.tolerance = 1.0e-3
        config.strain_basis = "P0"
        config.collider_basis = "pic"
        config.velocity_basis = "Q1"
        config.collider_velocity_mode = "forward"
        self.solver = SolverImplicitMPM(self.model, config=config)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self._body_pose = self.state_0.body_q.numpy().copy()
        self._body_velocity = np.zeros((self.model.body_count, 6), dtype=np.float32)
        self._set_tube_state(0.0)
        collider_count = self.solver.collider_body_index.shape[0]
        self.solver.setup_collider(
            body_q=self.state_0.body_q,
            collider_projection_threshold=[-0.25 * self.spacing] * collider_count,
            collider_friction=[0.0] * collider_count,
            collider_adhesion=[0.0] * collider_count,
        )
        self.solver.reset(self.state_0)
        self.solver.reset(self.state_1)
        # Store only real state arrays; State.assign also sees transient solver
        # scratch arrays after the first step and cannot safely restore a clone.
        self._rest = {
            name: wp.clone(getattr(self.state_0, name))
            for name in ("particle_q", "particle_qd", "body_q", "body_qd")
        }
        self._rest_mpm = {
            name: wp.clone(getattr(self.state_0.mpm, name))
            for name in (
                "particle_elastic_strain", "particle_transform", "particle_qd_grad",
                "particle_stress", "particle_Jp",
            )
        }
        self.surface = self.solver.create_particle_surface(
            voxel_size=float(surface_voxel),
            max_grid_cells=int(surface_max_cells),
            surface_method="particle_sdf", field_mode="sdf",
            particle_sdf_radius_scale=1.3, redistance_iterations=1,
            kernel_radius=max(2.5 * self.spacing, 1.5 * float(surface_voxel)),
            threshold=0.22, smooth_lambda=0.25, anisotropic=True,
            kernel_scale=0.5, anisotropy_ratio=4.0, anisotropy_scale=1.0,
            anisotropy_min_neighbors=6, anisotropy_binning=True,
            anisotropy_strength=0.8, field_smooth_iterations=1,
            field_smooth_radius=1, mesh_smooth_iterations=1,
        )


        self.mode = "Warp PBF with analytic glass contacts"
        # Infer the two analytic hollow vessels from the actual input meshes.
        # The tube is a cylindrical bore with a hemispherical bottom; the
        # beaker is a flat-bottom cylindrical shell. Both openings remain open.
        tv = np.asarray(tube_mesh[0], dtype=np.float64)
        tl = float(tv[:, 2].max())
        tr = np.linalg.norm(tv[np.abs(tv[:, 2]-tl)<1.0e-6, :2], axis=1)
        tube_r, tube_outer = float(tr.min()), float(tr.max())
        bv = np.asarray(beaker_mesh[0], dtype=np.float64)
        bc = bv[:, :2].mean(axis=0)
        bz = float(bv[:, 2].max())
        br = np.linalg.norm(bv[np.abs(bv[:, 2]-bz)<1.0e-6, :2]-bc, axis=1)
        beaker_r, beaker_outer = float(br.min()), float(br.max())
        beaker_thickness = beaker_outer-beaker_r
        floor = float(bv[:, 2].min())+beaker_thickness
        self._analytic_tube = wp.vec4(tube_r, tl, tube_outer-tube_r, 0.0)
        self._analytic_beaker = wp.vec4(float(bc[0]), float(bc[1]), floor, beaker_r)
        self._analytic_beaker_ext = wp.vec2(bz-floor, beaker_thickness)
        self._kernels = _build_pbf_kernels()
        self._grid = wp.HashGrid(64, 64, 64, device=self.model.device)
        self._lambdas = wp.zeros(self.particle_count, dtype=float, device=self.model.device)
        self._density_ratio = wp.zeros(self.particle_count, dtype=float, device=self.model.device)
        self._delta_position = wp.empty(self.particle_count, dtype=wp.vec3, device=self.model.device)
        self._velocity_temp = wp.empty_like(self.state_0.particle_qd)
        self._pbf_iterations = 6
        self._support = 2.0 * self.spacing
        # Discrete cubic-lattice normalization keeps rho=1000 at the requested
        # spacing while mass is exactly rho*spacing**3.
        h = self._support
        calibration = 0.0
        for x in range(-2, 3):
            for y in range(-2, 3):
                for z in range(-2, 3):
                    r2 = self.spacing**2 * (x*x + y*y + z*z)
                    if r2 < h*h:
                        calibration += self.spacing**3 * (315.0 / (64.0 * np.pi * h**9)) * (h*h-r2)**3
        self._normalization = 1.0 / calibration
        self._weight_coefficient = (315.0 / (64.0 * np.pi * h**9)) * self._normalization
        self._gradient_coefficient = (-45.0 / (np.pi * h**6)) * self._normalization
        self._kinematic_extent = float(np.linalg.norm(np.asarray(tube_mesh[0]), axis=1).max())
        self._kinematic_viscosity = float(viscosity / density)
        self.last_substeps = self.substeps
        self._total_substeps = 0

    @property
    def average_substeps(self):
        return self._total_substeps / max(1, self._frame_index)

    @property
    def density_ratios(self):
        return self._density_ratio.numpy().copy()

    @property
    def time(self):
        return self._frame_index * self.dt

    def _sample_pose(self, t):
        np = self.np
        p, q = self.pose_fn(float(t))
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        if p.shape != (3,) or q.shape != (4,) or not np.isfinite(p).all() or not np.isfinite(q).all():
            raise ValueError("pose_fn must return finite position(3), quaternion_xyzw(4).")
        length = np.linalg.norm(q)
        if length < 1.0e-12:
            raise ValueError("pose_fn returned a zero quaternion.")
        return p, q / length

    def _pose_and_twist(self, t):
        np = self.np
        p, q = self._sample_pose(t)
        p_next, q_next = self._sample_pose(t + self.sub_dt)
        if np.dot(q, q_next) < 0.0:
            q_next = -q_next
        # q_next * conjugate(q) is the angular increment in world coordinates.
        a, b = q_next[:3], -q[:3]
        delta_xyz = q_next[3] * b + q[3] * a + np.cross(a, b)
        delta_w = np.clip(q_next[3] * q[3] - np.dot(a, b), -1.0, 1.0)
        sine = np.linalg.norm(delta_xyz)
        if sine < 1.0e-12:
            angular = np.zeros(3)
        else:
            angular = delta_xyz * (2.0 * np.arctan2(sine, delta_w) / (sine * self.sub_dt))
        velocity = np.concatenate(((p_next - p) / self.sub_dt, angular))
        return np.concatenate((p, q)).astype(np.float32), velocity.astype(np.float32)

    def _set_tube_state(self, t):
        pose, velocity = self._pose_and_twist(t)
        self._body_pose[self.tube_body] = pose
        self._body_velocity[self.tube_body] = velocity
        # Newton uses [linear_xyz, angular_xyz], not angular first.
        for state in (self.state_0, self.state_1):
            state.body_q.assign(self._body_pose)
            state.body_qd.assign(self._body_velocity)

    def step_frame(self):
        import math
        np, wp = self.np, self.wp
        start = self.time
        velocities = self.state_0.particle_qd.numpy()
        speed = float(np.linalg.norm(velocities, axis=1).max())
        pose, twist = self._pose_and_twist(start + 0.5 * self.dt)
        boundary_speed = float(np.linalg.norm(twist[:3]) + np.linalg.norm(twist[3:]) * self._kinematic_extent)
        speed = max(speed, boundary_speed) + 9.81 * self.dt
        steps = max(self.substeps, int(math.ceil(speed * self.dt / (0.35 * self.spacing))))
        if steps > 160:
            raise RuntimeError(f"Fluid speed requires {steps} substeps; unstable state.")
        sub_dt = self.dt / steps
        self.last_substeps = steps
        self._total_substeps += steps
        predict, density, pressure, apply, update_velocity, viscosity, project = self._kernels
        tube, beaker, beaker_ext = self._analytic_tube, self._analytic_beaker, self._analytic_beaker_ext
        h = self._support
        volume = self.spacing**3
        for substep in range(steps):
            self._set_tube_state(start + (substep + 1) * sub_dt)
            wp.launch(predict, dim=self.particle_count,
                      inputs=[self.state_0.particle_q, self.state_0.particle_qd,
                              self.state_1.particle_q, self.state_1.particle_qd,
                              sub_dt, 0.25 * self.spacing], device=self.model.device)
            # Endpoint geometry, without extrapolating the tube a second time.
            # Forward-mode native projection is well-defined for dt=0.
            wp.launch(project, dim=self.particle_count,
                      inputs=[self.state_1.particle_q, tube, beaker, beaker_ext,
                              self.state_1.body_q, 0.25*self.spacing], device=self.model.device)
            for iteration in range(self._pbf_iterations):
                self._grid.build(self.state_1.particle_q, h)
                shared = [h, volume, self._weight_coefficient, self._gradient_coefficient,
                          self._normalization, tube, beaker, beaker_ext, self.state_1.body_q]
                wp.launch(density, dim=self.particle_count,
                          inputs=[self.state_1.particle_q, self._grid.id, *shared,
                                  self._lambdas, self._density_ratio], device=self.model.device)
                wp.launch(pressure, dim=self.particle_count,
                          inputs=[self.state_1.particle_q, self._grid.id, h, volume,
                                  self._gradient_coefficient, self._normalization, tube, beaker,
                                  beaker_ext, self.state_1.body_q, self._lambdas,
                                  self._delta_position, 0.2 * self.spacing], device=self.model.device)
                wp.launch(apply, dim=self.particle_count,
                          inputs=[self.state_1.particle_q, self._delta_position, 0.25*self.spacing],
                          device=self.model.device)
                wp.launch(project, dim=self.particle_count,
                          inputs=[self.state_1.particle_q, tube, beaker, beaker_ext,
                                  self.state_1.body_q, 0.25*self.spacing], device=self.model.device)
            wp.launch(update_velocity, dim=self.particle_count,
                      inputs=[self.state_0.particle_q, self.state_1.particle_q,
                              self.state_1.particle_qd, sub_dt], device=self.model.device)
            if self._kinematic_viscosity > 0.0:
                self._grid.build(self.state_1.particle_q, h)
                wp.launch(viscosity, dim=self.particle_count,
                          inputs=[self.state_1.particle_q, self.state_1.particle_qd,
                                  self._velocity_temp, self._grid.id, h, volume,
                                  self._weight_coefficient,
                                  min(0.05, 6.0*self._kinematic_viscosity*sub_dt/(h*h))],
                          device=self.model.device)
                wp.copy(self.state_1.particle_qd, self._velocity_temp)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self._frame_index += 1

    def positions(self):
        return self.state_0.particle_q.numpy().copy()

    def extract_surface(self):
        np = self.np
        mesh = self.solver.extract_particle_surface(
            self.state_0, self.surface, compute_normals=True
        )
        vertices, indices, normals = mesh.to_arrays()
        if vertices is None or indices is None:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty(0, dtype=np.int32),
                np.empty((0, 3), dtype=np.float32),
            )
        points = vertices.numpy().astype(np.float32, copy=True)
        faces = indices.numpy().astype(np.int32, copy=True).reshape(-1)
        if normals is not None:
            normal_values = normals.numpy().astype(np.float32, copy=True)
        else:
            normal_values = np.zeros_like(points)
            triangles = faces.reshape(-1, 3)
            cross = np.cross(
                points[triangles[:, 1]] - points[triangles[:, 0]],
                points[triangles[:, 2]] - points[triangles[:, 0]],
            )
            for corner in range(3):
                np.add.at(normal_values, triangles[:, corner], cross)
            lengths = np.linalg.norm(normal_values, axis=1)
            normal_values /= np.maximum(lengths[:, None], 1.0e-12)
        return points, faces, normal_values

    def reset(self):
        """Restore the original fixed particles and clear MPM warm-start history."""
        for state in (self.state_0, self.state_1):
            for name, rest in self._rest.items():
                getattr(state, name).assign(rest)
            for name, rest in self._rest_mpm.items():
                getattr(state.mpm, name).assign(rest)
            state.clear_forces()
        self._frame_index = 0
        self._total_substeps = 0
        self.solver.reset(self.state_0)
        self.solver.reset(self.state_1)
        self._set_tube_state(0.0)

