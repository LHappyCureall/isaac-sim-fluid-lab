"""Feedback dosing controller; pure NumPy, no particle/state modifications.

The geometry's existing motion is a path parameterized by ``source_time``.
Call update once after each physics frame. Pose queries are pure functions of
committed commands; repeated or finite-difference queries never advance control.
"""
import bisect
import math
import numpy as np


class TargetPourController:
    def __init__(self, geometry, target_ml=30.0, spacing=0.0035):
        self.geometry = geometry
        self.spacing = float(spacing)
        if not math.isfinite(self.spacing) or self.spacing <= 0:
            raise ValueError("spacing must be positive and finite")
        self.particle_ml = self.spacing ** 3 * 1.0e6
        self.upright_source = float(geometry.MOTION_KNOTS[2][0])
        # Reach 78 degrees at the original path's bounded speed, then meter
        # slowly through the narrow angle range where this tube starts flowing.
        low = float(geometry.MOTION_KNOTS[3][0])
        high = float(geometry.MOTION_KNOTS[4][0])
        for _ in range(48):
            middle = 0.5 * (low + high)
            if math.degrees(geometry.mouth_pose(middle)[1]) < 78.0:
                low = middle
            else:
                high = middle
        self.pretilt_source = 0.5 * (low + high)
        self.max_source = float(geometry.MOTION_KNOTS[4][0])
        self.max_speed = 1.0
        self.max_acceleration = 3.0
        self.reset(target_ml)

    def reset(self, target_ml=None):
        if target_ml is not None:
            self.target_ml = float(target_ml)
        if not math.isfinite(self.target_ml) or not 0 < self.target_ml <= 45:
            raise ValueError("target_ml must be in (0, 45] for this 51 mL tube")
        self.phase = "prepare"
        self.done = False
        self._segments = [(0.0, 0.0, 0.0, 0.0, 1.0 / 60.0, self.pretilt_source)]
        self._times = [0.0]
        self._last_t = None
        self._last_points = None
        self._last_released = 0.0
        self._flow = 0.0
        self._settle_since = None
        self._stable_since = None
        self._last_received = 0.0
        self._last_stop_source = None
        self._trim_count = 0
        self.status = dict(phase=self.phase, done=False, target_ml=self.target_ml,
                           received_ml=0.0, released_ml=0.0, inflight_ml=0.0,
                           flow_ml_s=0.0, predicted_final_ml=0.0, source_time=0.0,
                           tilt_deg=0.0, error_ml=-self.target_ml)
        return dict(self.status)

    @staticmethod
    def _evaluate(segment, t):
        t0, s0, v0, acceleration, horizon, goal = segment
        elapsed = max(0.0, float(t) - t0)
        h = min(elapsed, horizon)
        s = s0 + v0 * h + 0.5 * acceleration * h * h
        velocity = v0 + acceleration * h
        s += velocity * max(0.0, elapsed - horizon)
        direction = 1.0 if goal >= s0 else -1.0
        if direction * (s - goal) >= 0.0:
            return goal, 0.0
        return s, velocity

    def source_time(self, t):
        index = max(0, bisect.bisect_right(self._times, float(t)) - 1)
        return float(np.clip(self._evaluate(self._segments[index], t)[0], 0.0, self.max_source))

    def tube_pose(self, t):
        return self.geometry.tube_pose(self.source_time(t))

    def _command(self, t, goal, max_rate, dt):
        s, v = self._evaluate(self._segments[-1], t)
        distance = goal - s
        desired = math.copysign(min(float(max_rate), math.sqrt(2 * self.max_acceleration * abs(distance))), distance) if abs(distance) > 1e-9 else 0.0
        next_v = float(np.clip(desired, v-self.max_acceleration*dt, v+self.max_acceleration*dt))
        acceleration = (next_v-v) / dt
        segment = (float(t), s, v, acceleration, dt, float(goal))
        if self._times[-1] == float(t):
            self._segments[-1] = segment
        else:
            self._times.append(float(t))
            self._segments.append(segment)

    def update(self, t, positions, labels, velocities=None):
        """Observe fixed particles and commit one continuous motion command.

        Volumes refer only to particles initially in the tube. In-flight water
        is counted before it reaches the beaker. A ballistic rim-crossing test
        rejects obvious misses; its estimate is used only to lead the motion.
        """
        t = float(t)
        if self._last_t is not None and t <= self._last_t:
            if abs(t-self._last_t) < 1e-9:
                return dict(self.status)
            raise ValueError("physical time must increase; reset before restarting")
        points = np.asarray(positions, dtype=np.float64)
        groups = np.asarray(labels)
        if points.ndim != 2 or points.shape[1] != 3 or groups.shape != (len(points),) or not np.isfinite(points).all():
            raise ValueError("positions must be finite Nx3 with matching fixed labels")
        dt = t-self._last_t if self._last_t is not None else 1.0/60.0
        source = self.source_time(t)
        g = self.geometry
        mouth, angle = g.mouth_pose(source)
        origin, _ = g.tube_pose(source)
        local = (points-np.asarray(origin)) @ g.rotation_y(angle)
        tube = g._cavity_mask(local)
        center = np.asarray(g.BEAKER_CENTER)
        delta = points-center
        beaker = (np.sum(delta[:, :2]**2, axis=1) <= g.BEAKER_RADIUS**2) & (delta[:, 2]>=0) & (delta[:, 2]<=g.BEAKER_HEIGHT)
        initial_tube = groups == 1
        free = initial_tube & ~(tube | beaker)
        airborne = free & (points[:, 2] > g.TABLE_TOP_Z+self.spacing)
        if velocities is None:
            velocity = (points-self._last_points)/dt if self._last_points is not None else np.zeros_like(points)
        else:
            velocity = np.asarray(velocities, dtype=np.float64)
            if velocity.shape != points.shape or not np.isfinite(velocity).all():
                raise ValueError("velocities must be finite and match positions")
        rim_z = center[2]+g.BEAKER_HEIGHT
        discriminant = velocity[:, 2]**2+2*9.81*np.maximum(points[:, 2]-rim_z, 0)
        fall_time = (velocity[:, 2]+np.sqrt(discriminant))/9.81
        impact_xy = points[:, :2]+velocity[:, :2]*fall_time[:, None]
        catch = (np.sum((impact_xy-center[:2])**2, axis=1) <= (g.BEAKER_RADIUS+0.5*self.spacing)**2)
        # A center already below the rim may enter the interior next frame.
        catch |= (points[:, 2] <= rim_z) & (np.sum(delta[:, :2]**2, axis=1) <= g.BEAKER_RADIUS**2)
        received = float(np.count_nonzero(initial_tube & beaker)*self.particle_ml)
        released = float(np.count_nonzero(initial_tube & ~tube)*self.particle_ml)
        inflight = float(np.count_nonzero(airborne & catch)*self.particle_ml)
        instantaneous_flow = max(0.0, (released-self._last_released)/dt)
        blend = 1-math.exp(-dt/0.12)
        self._flow += blend*(instantaneous_flow-self._flow)
        predicted_final = received+inflight+self._flow*0.20
        if received != self._last_received:
            self._stable_since = t
        elif self._stable_since is None:
            self._stable_since = t
        tilt = math.degrees(float(angle))
        if self.phase == "prepare" and source >= self.pretilt_source-1.0e-5:
            self.phase = "pour"
        if self.phase == "pour" and predicted_final >= self.target_ml-0.20 and released > 1.0:
            self.phase = "return"
            self._last_stop_source = source
        if self.phase == "return" and source <= self.upright_source+1.0e-5:
            self.phase = "settle"
            self._settle_since = t
        if self.phase == "settle" and t-self._settle_since >= 1.2 and inflight <= 0.1 and t-self._stable_since >= 0.5:
            if received < self.target_ml-1.0 and self._trim_count < 2:
                self.phase = "trim_approach"
                self._trim_count += 1
            else:
                self.phase, self.done = "done", True
        if self.phase == "trim_approach" and source >= self._last_stop_source-0.02:
            self.phase = "pour"
        if self.phase == "prepare":
            self._command(t, self.pretilt_source, 1.0, dt)
        elif self.phase == "trim_approach":
            self._command(t, max(self.pretilt_source, self._last_stop_source-0.02), 1.0, dt)
        elif self.phase == "pour":
            remaining = max(0.0, self.target_ml-(received+inflight))
            rate = 0.05 if tilt < 80 else 0.025
            if remaining < 8:
                rate = min(rate, 0.012)
            self._command(t, self.max_source, rate, dt)
        elif self.phase == "return":
            self._command(t, self.upright_source, 1.0, dt)
        else:
            self._command(t, self.upright_source, 1.0, dt)
        count = int(np.count_nonzero(initial_tube))
        self.status = dict(phase=self.phase, done=self.done, target_ml=self.target_ml,
            received_ml=received, released_ml=released, inflight_ml=inflight,
            flow_ml_s=float(self._flow), predicted_final_ml=float(predicted_final),
            source_time=source, tilt_deg=tilt, error_ml=received-self.target_ml,
            particle_count=len(points), tube_remaining_ml=(count*self.particle_ml-released),
            spilled_ml=float(np.count_nonzero(free & ~airborne)*self.particle_ml),
            trim_count=self._trim_count)
        self._last_t = t
        self._last_points = points.copy()
        self._last_released = released
        self._last_received = received
        return dict(self.status)
