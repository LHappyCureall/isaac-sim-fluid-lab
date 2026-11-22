"""Independent full-R shortcut pilot; development only, not held-out data."""
import argparse
import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / 'assets/robot_testtube_pour/scripts'
spec = importlib.util.spec_from_file_location('direct_pilot_shared', ROOT / 'train_target_volume_policy.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--targets', nargs='+', type=int, default=[20, 30, 40])
    parser.add_argument('--seed', type=int, default=80101)
    parser.add_argument('--params', type=Path)
    parser.add_argument('--randomize', action='store_true')
    parser.add_argument('--diagnostics', action='store_true', help='Read-only first escape/landing particle evidence.')
    parser.add_argument('--geometry-cache', type=Path, help='Explicit independently prepared fixed-mouth robot path cache.')
    parser.add_argument('--geometry-cache-sha256', help='Expected raw SHA256 for the selected robot path cache.')
    parser.add_argument('--lift-first', action='store_true', help='Use the separately prepared upright-lift fixed-mouth geometry.')
    parser.add_argument('--control-version', type=int, choices=(1, 2), default=1,
                        help='Version 2 is the independent lift return-planning controller.')
    args = parser.parse_args()
    if any(t not in (20, 30, 40) for t in args.targets):
        raise ValueError('Choose 20/30/40mL.')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'protocol.json').exists():
        raise RuntimeError('Use a fresh direct pilot directory; preserve development failures.')
    with (args.output / 'worker.json').open('x', encoding='utf-8') as stream:
        json.dump({'state': 'entry_started', 'worker_pid': os.getpid(),
            'started_at_utc': datetime.now(timezone.utc).isoformat(), 'command': sys.argv,
            'entry_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, stream, indent=2)
    if (args.geometry_cache is None) != (args.geometry_cache_sha256 is None):
        raise ValueError('Select a fixed-mouth cache and its SHA256 together.')
    if args.lift_first and args.geometry_cache is None:
        raise ValueError('Upright-lift geometry requires its explicit cache and SHA256.')
    if args.control_version == 2 and not args.lift_first:
        raise ValueError('The new return planner is bound to the upright-lift geometry.')
    versions = shared.configure_sdk()
    if args.geometry_cache is None:
        import direct_fast_geometry as geometry
    elif args.lift_first:
        import mouth_pivot_lift_geometry as geometry_module
        geometry = geometry_module.load_geometry(args.geometry_cache, args.geometry_cache_sha256)
    else:
        import mouth_pivot_fast_geometry as geometry_module
        geometry = geometry_module.load_geometry(args.geometry_cache, args.geometry_cache_sha256)
    if args.geometry_cache is None:
        import direct_fast_control as control
    elif args.lift_first and args.control_version == 2:
        import mouth_pivot_lift_control_v2 as control
    elif args.lift_first:
        import mouth_pivot_lift_control as control
    else:
        import mouth_pivot_control as control
    import direct_training_env as collector
    if args.diagnostics:
        import direct_diagnostic_env as diagnostic
    parameters = json.loads(args.params.read_text(encoding='utf-8')) if args.params else None
    policy = control.DirectPolicy(parameters)
    dependencies = [ROOT / 'train_target_volume_policy.py', Path(__file__)]
    dependencies += [SCRIPTS / name for name in ('geometry.py', 'liquid.py', 'target_control.py',
        'pour_training_env.py', 'fast_geometry.py', 'fast_control.py', 'fast_training_env.py',
        'direct_fast_geometry.py', 'direct_fast_control.py', 'direct_training_env.py')]
    if args.diagnostics:
        dependencies.append(SCRIPTS / 'direct_diagnostic_env.py')
    if args.geometry_cache is not None:
        dependencies += [Path(geometry.__file__), Path(control.__file__),
                         ROOT / 'tools/training/prepare_mouth_pivot_path.py']
    if args.lift_first:
        dependencies += [SCRIPTS / 'mouth_pivot_fast_geometry.py',
                         ROOT / 'tools/training/prepare_mouth_pivot_lift_path.py']
    hashes = {}
    for source in dependencies:
        relative = source.relative_to(ROOT).as_posix()
        content = source.read_bytes()
        hashes[relative] = hashlib.sha256(content).hexdigest()
        destination = args.output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    cache_content = Path(geometry.CACHE_FILE).read_bytes()
    if hashlib.sha256(cache_content).hexdigest() != geometry.CACHE_SHA256:
        raise RuntimeError('The selected robot geometry cache changed before freezing.')
    cache_snapshot = args.output / 'input_snapshot/path_cache.npz'
    cache_snapshot.parent.mkdir(parents=True, exist_ok=True)
    cache_snapshot.write_bytes(cache_content)
    extra_cache_inputs = {}
    parameter_input = None
    if args.params is not None:
        parameter_raw = args.params.read_bytes()
        if json.loads(parameter_raw) != parameters:
            raise RuntimeError('Parameter input changed during setup.')
        parameter_snapshot = args.output / 'input_snapshot/policy_parameters_input.json'
        parameter_snapshot.write_bytes(parameter_raw)
        parameter_input = {'path': str(args.params.resolve()),
            'sha256': hashlib.sha256(parameter_raw).hexdigest(),
            'snapshot': 'input_snapshot/policy_parameters_input.json'}
    if args.geometry_cache is not None:
        default_cache = Path(geometry_module.DEFAULT_CACHE_FILE)
        raw_default = default_cache.read_bytes()
        if hashlib.sha256(raw_default).hexdigest() != geometry_module.DEFAULT_CACHE_SHA256:
            raise RuntimeError('The module default import cache changed.')
        (args.output / 'input_snapshot/default_import_path_cache.npz').write_bytes(raw_default)
        extra_cache_inputs['default_import_cache'] = {'path': str(default_cache),
            'sha256': geometry_module.DEFAULT_CACHE_SHA256,
            'snapshot': 'input_snapshot/default_import_path_cache.npz'}
        if args.lift_first:
            for role, source, expected in (
                    ('base_path', geometry_module.BASE_CACHE_FILE, geometry_module.BASE_CACHE_SHA256),
                    ('base_default_import', geometry_module.BASE_IMPORT_CACHE_FILE, geometry_module.BASE_IMPORT_CACHE_SHA256)):
                content = Path(source).read_bytes()
                if hashlib.sha256(content).hexdigest() != expected:
                    raise RuntimeError('A transitive lift geometry cache changed: ' + role)
                snapshot = 'input_snapshot/' + role + '_cache.npz'
                (args.output / snapshot).write_bytes(content)
                extra_cache_inputs[role] = {'path': str(source), 'sha256': expected, 'snapshot': snapshot}
    protocol = {'purpose': 'direct_path_development_pilot_not_heldout', 'versions': versions,
        'sources': hashes, 'cache_sha256': geometry.CACHE_SHA256,
        'geometry_kind': ('C2_upright_lift_fixed_mouth_offline_IK' if args.lift_first
                          else 'C2_fixed_mouth_offline_IK' if args.geometry_cache is not None
                          else 'C2_joint_shortcut_with_full_R_tube_containment'),
        'effective_geometry_file': Path(geometry.__file__).relative_to(ROOT).as_posix(),
        'effective_controller_file': Path(control.__file__).relative_to(ROOT).as_posix(),
        'robot_path_cache_snapshot': 'input_snapshot/path_cache.npz',
        'additional_cache_inputs': extra_cache_inputs,
        'parameter_input': parameter_input,
        'targets_ml': args.targets, 'seed': args.seed, 'randomized': args.randomize,
        'policy_parameters': policy.params, 'deadline_s': 5.0,
        'observation_end_physical_time_s': 7.0, 'physical_dt_s': 1/60,
        'pressure_iterations': 6, 'minimum_substeps': 4, 'cuda_graph': False,
        'liquid_solver_modified': False,
        'joint_acceleration_simulation_cap_rad_s2': 25.0,
        'path_coordinate_acceleration_cap': None,
        'coordinate_rate_switch_only_where_physical_joint_derivatives_zero': True}
    protocol['read_only_particle_escape_diagnostics'] = args.diagnostics
    shared.write_json(args.output / 'protocol.json', protocol)
    policy.save(args.output / 'parameters.json', metadata={'purpose': protocol['purpose']})
    environment_class = diagnostic.DiagnosticDirectTrainingEnv if args.diagnostics else collector.DirectTrainingEnv
    env = environment_class(geometry=geometry, control=control)
    results = []
    start = time.perf_counter()
    for target in args.targets:
        def progress(event):
            shared.write_json(args.output / 'progress.json', {'state': 'running',
                'completed': len(results), 'wall_seconds': time.perf_counter() - start, **event})
        result = env.run_episode(target, args.seed, policy=policy, randomize=args.randomize,
            record_path=args.output / f'target_{target}.steps.jsonl.gz', progress=progress)
        extra = {'motion_metrics': dict(env.fast_metrics), 'controller': dict(env.last_controller.status)}
        try:
            extra['post_observation'] = (env.post_observe(args.output / f'target_{target}.post.jsonl.gz')
                                        if result['valid_physics'] else None)
        except Exception as error:
            extra['post_observation'] = {'passed': False, 'exception': type(error).__name__ + ': ' + str(error)}
        extra['post_motion_metrics'] = dict(env.fast_metrics)
        results.append({'result': result, **extra})
        shared.write_json(args.output / 'results.json', {'complete': False, 'results': results})
        print('DIRECT PILOT ' + json.dumps({'target': target, 'received': result['received_ml'],
            'error_ml': result['error_ml'], 'time_s': result['physical_time_s'],
            'complete': result['complete'], 'valid': result['valid_physics'],
            'spill_ml': result['max_spilled_ml'], 'error': result['error']}), flush=True)
    for relative, expected in hashes.items():
        if shared._sha(ROOT / relative) != expected:
            raise RuntimeError('An active direct pilot source changed: ' + relative)
    if shared._sha(geometry.CACHE_FILE) != geometry.CACHE_SHA256 or shared._sha(cache_snapshot) != geometry.CACHE_SHA256:
        raise RuntimeError('The active robot geometry cache or its snapshot changed.')
    for entry in extra_cache_inputs.values():
        if shared._sha(entry['path']) != entry['sha256'] or shared._sha(args.output / entry['snapshot']) != entry['sha256']:
            raise RuntimeError('An additional geometry import cache changed.')
    if parameter_input is not None:
        if shared._sha(parameter_input['path']) != parameter_input['sha256'] or shared._sha(args.output / parameter_input['snapshot']) != parameter_input['sha256']:
            raise RuntimeError('The parameter input or its frozen snapshot changed.')
    shared.write_json(args.output / 'results.json', {'complete': True, 'results': results,
        'purpose': protocol['purpose'], 'wall_seconds': time.perf_counter() - start})
    shared.write_json(args.output / 'progress.json', {'state': 'complete', 'completed': len(results)})
    worker = json.loads((args.output / 'worker.json').read_text(encoding='utf-8'))
    shared.write_json(args.output / 'worker.json', {**worker, 'state': 'complete',
        'finished_at_utc': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
