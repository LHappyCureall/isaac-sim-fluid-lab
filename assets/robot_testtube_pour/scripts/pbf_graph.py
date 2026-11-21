"""Optional CUDA graph dispatch for the existing PourFluid PBF substep.

This only batches the original GPU operations. Physical frame duration,
adaptive substeps, six pressure iterations, contacts, viscosity and host tube
pose samples are unchanged. Callers must validate numerical equivalence on
their installed Warp/CUDA build before enabling this for experiment data.
"""

from collections import OrderedDict
import hashlib
import inspect
import math
from types import MethodType


ORIGINAL_STEP_SOURCE_SHA256 = "7f30ecf6d8dac8672615b32433fdcbce46a243008adae4abad4a291bbbeec103"


def _step_source_hash(step):
    try:
        source = inspect.getsource(step)
    except (OSError, TypeError) as error:
        raise RuntimeError("Cannot inspect the original PourFluid.step_frame; graph dispatch is disabled.") from error
    normalized = source.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class PBFGraphStep:
    """Bound instance step with a small pointer- and parameter-sensitive cache."""

    def __init__(self, fluid, max_graphs=128):
        self.fluid = fluid
        self.original_step_frame = fluid.step_frame
        self.original_step_source_sha256 = _step_source_hash(self.original_step_frame)
        if self.original_step_source_sha256 != ORIGINAL_STEP_SOURCE_SHA256:
            raise RuntimeError("PourFluid.step_frame source changed; graph dispatch requires a fresh "
                               "operation-order review and numerical equivalence test. "
                               f"Expected {ORIGINAL_STEP_SOURCE_SHA256}, got {self.original_step_source_sha256}.")
        self.max_graphs = int(max_graphs)
        if self.max_graphs < 2:
            raise ValueError("max_graphs must be at least two for double-buffered states.")
        wp = fluid.wp
        device = wp.get_device(fluid.model.device)
        if not device.is_cuda:
            raise ValueError("PBF CUDA graph acceleration requires a CUDA device.")
        if fluid.substeps < 4 or fluid._pbf_iterations != 6:
            raise ValueError("Expected the validated >=4 substep, six-iteration PBF configuration.")
        self.device = device
        self.graphs = OrderedDict()
        self.captures = 0
        self.launches = 0
        self.evictions = 0
        self._viscosity = float(fluid._kinematic_viscosity)
        # The bundled Warp implementation documents reserve() before capture;
        # HashGrid.build is supported with apic=False and reuses this storage.
        fluid._grid.reserve(fluid.particle_count)
        modules = {id(kernel.module): kernel.module for kernel in fluid._kernels}
        for module in modules.values():
            wp.load_module(module=module, device=self.device)

    @staticmethod
    def _pointer(array):
        return (id(array), int(array.ptr))

    def _key(self, sub_dt):
        f = self.fluid
        arrays = (f.state_0.particle_q, f.state_0.particle_qd,
                  f.state_1.particle_q, f.state_1.particle_qd,
                  f.state_1.body_q, f._lambdas, f._density_ratio,
                  f._delta_position, f._velocity_temp)
        return (float(sub_dt), tuple(self._pointer(array) for array in arrays),
                int(f._grid.id), int(f.particle_count), int(f._pbf_iterations),
                tuple(id(kernel) for kernel in f._kernels),
                float(f._support), float(f.spacing), float(f._weight_coefficient),
                float(f._gradient_coefficient), float(f._normalization),
                float(f._kinematic_viscosity),
                tuple(float(x) for x in f._analytic_tube),
                tuple(float(x) for x in f._analytic_beaker),
                tuple(float(x) for x in f._analytic_beaker_ext))

    def _dispatch_substep(self, sub_dt):
        """Verbatim order and scalar expressions from PourFluid.step_frame."""
        f, wp = self.fluid, self.fluid.wp
        predict, density, pressure, apply, update_velocity, viscosity, project = f._kernels
        tube, beaker, beaker_ext = f._analytic_tube, f._analytic_beaker, f._analytic_beaker_ext
        h, volume = f._support, f.spacing ** 3
        wp.launch(predict, dim=f.particle_count,
                  inputs=[f.state_0.particle_q, f.state_0.particle_qd,
                          f.state_1.particle_q, f.state_1.particle_qd,
                          sub_dt, 0.25 * f.spacing], device=f.model.device)
        wp.launch(project, dim=f.particle_count,
                  inputs=[f.state_1.particle_q, tube, beaker, beaker_ext,
                          f.state_1.body_q, 0.25 * f.spacing], device=f.model.device)
        for iteration in range(f._pbf_iterations):
            f._grid.build(f.state_1.particle_q, h)
            shared = [h, volume, f._weight_coefficient, f._gradient_coefficient,
                      f._normalization, tube, beaker, beaker_ext, f.state_1.body_q]
            wp.launch(density, dim=f.particle_count,
                      inputs=[f.state_1.particle_q, f._grid.id, *shared,
                              f._lambdas, f._density_ratio], device=f.model.device)
            wp.launch(pressure, dim=f.particle_count,
                      inputs=[f.state_1.particle_q, f._grid.id, h, volume,
                              f._gradient_coefficient, f._normalization, tube, beaker,
                              beaker_ext, f.state_1.body_q, f._lambdas,
                              f._delta_position, 0.2 * f.spacing], device=f.model.device)
            wp.launch(apply, dim=f.particle_count,
                      inputs=[f.state_1.particle_q, f._delta_position, 0.25 * f.spacing],
                      device=f.model.device)
            wp.launch(project, dim=f.particle_count,
                      inputs=[f.state_1.particle_q, tube, beaker, beaker_ext,
                              f.state_1.body_q, 0.25 * f.spacing], device=f.model.device)
        wp.launch(update_velocity, dim=f.particle_count,
                  inputs=[f.state_0.particle_q, f.state_1.particle_q,
                          f.state_1.particle_qd, sub_dt], device=f.model.device)
        if f._kinematic_viscosity > 0.0:
            f._grid.build(f.state_1.particle_q, h)
            wp.launch(viscosity, dim=f.particle_count,
                      inputs=[f.state_1.particle_q, f.state_1.particle_qd,
                              f._velocity_temp, f._grid.id, h, volume,
                              f._weight_coefficient,
                              min(0.05, 6.0 * f._kinematic_viscosity * sub_dt / (h * h))],
                      device=f.model.device)
            wp.copy(f.state_1.particle_qd, f._velocity_temp)

    def step_frame(self):
        f, np, wp = self.fluid, self.fluid.np, self.fluid.wp
        start = f.time
        velocities = f.state_0.particle_qd.numpy()
        speed = float(np.linalg.norm(velocities, axis=1).max())
        pose, twist = f._pose_and_twist(start + 0.5 * f.dt)
        boundary_speed = float(np.linalg.norm(twist[:3]) + np.linalg.norm(twist[3:]) * f._kinematic_extent)
        speed = max(speed, boundary_speed) + 9.81 * f.dt
        steps = max(f.substeps, int(math.ceil(speed * f.dt / (0.35 * f.spacing))))
        if steps > 160:
            raise RuntimeError(f"Fluid speed requires {steps} substeps; unstable state.")
        sub_dt = f.dt / steps
        f.last_substeps = steps
        f._total_substeps += steps
        if self._viscosity != float(f._kinematic_viscosity):
            # Episode randomization changes this scalar continuously; retaining
            # all old viscosity graphs would accumulate unused CUDA resources.
            self.clear()
            self._viscosity = float(f._kinematic_viscosity)
        for substep in range(steps):
            # Pose/twist copies remain outside capture and run at every original
            # adaptive endpoint. Graph launches read the current body_q values.
            f._set_tube_state(start + (substep + 1) * sub_dt)
            key = self._key(sub_dt)
            graph = self.graphs.get(key)
            if graph is None:
                with wp.ScopedCapture(device=self.device, apic=False,
                                      force_module_load=False) as capture:
                    self._dispatch_substep(sub_dt)
                graph = capture.graph
                self.graphs[key] = graph
                self.captures += 1
                if len(self.graphs) > self.max_graphs:
                    self.graphs.popitem(last=False)
                    self.evictions += 1
            else:
                self.graphs.move_to_end(key)
            wp.capture_launch(graph)
            self.launches += 1
            f.state_0, f.state_1 = f.state_1, f.state_0
        f._frame_index += 1

    def clear(self):
        self.graphs.clear()

    def metrics(self):
        return {"dispatch": "cuda_graph_per_original_pbf_substep",
                "captures": self.captures, "launches": self.launches,
                "cached_graphs": len(self.graphs), "cache_limit": self.max_graphs,
                "evictions": self.evictions, "physical_solver_modified": False,
                "original_step_source_sha256": self.original_step_source_sha256}


def install_graph_step(fluid, *, max_graphs=128):
    """Install on one existing instance and return its graph/cache controller.

    Installation never silently falls back when a driver cannot capture the
    hash-grid operation. A failed capture raises, and that episode must be
    reset before retrying the original solver.
    """
    previous = getattr(fluid, "_pbf_graph_step", None)
    if previous is not None:
        return previous
    controller = PBFGraphStep(fluid, max_graphs=max_graphs)
    fluid._pbf_graph_step = controller
    fluid.step_frame = MethodType(lambda instance: instance._pbf_graph_step.step_frame(), fluid)
    return controller


def uninstall_graph_step(fluid):
    """Restore original dispatch; callers reset the fluid before a new rollout."""
    controller = getattr(fluid, "_pbf_graph_step", None)
    if controller is not None:
        fluid.step_frame = controller.original_step_frame
        controller.clear()
        del fluid._pbf_graph_step
    return controller
