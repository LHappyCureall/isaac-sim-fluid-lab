"""Engineering feasibility only: true five-second actions plus late-drop checks."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / 'assets/robot_testtube_pour/scripts'
spec = importlib.util.spec_from_file_location('fast_pilot_shared', ROOT / 'train_target_volume_policy.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--targets', nargs='+', type=int, default=[20, 30, 40])
    parser.add_argument('--seed', type=int, default=60101)
    parser.add_argument('--params', type=Path)
    parser.add_argument('--randomize', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'protocol.json').exists():
        raise RuntimeError('Use a new engineering pilot directory; never overwrite evidence.')
    versions = shared.configure_sdk()
    import fast_geometry
    import fast_control
    import fast_training_env
    policy = fast_control.FastPolicy(json.loads(args.params.read_text()) if args.params else None)
    dependencies = [ROOT / 'train_target_volume_policy.py', Path(__file__)]
    dependencies += [SCRIPTS / name for name in ('geometry.py', 'liquid.py', 'target_control.py',
        'pour_training_env.py', 'pbf_graph.py', 'fast_geometry.py', 'fast_control.py', 'fast_training_env.py')]
    hashes = {}
    for source in dependencies:
        relative = str(source.relative_to(ROOT)).replace('\\', '/')
        content = source.read_bytes()
        hashes[relative] = hashlib.sha256(content).hexdigest()
        destination = args.output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    declaration = {'purpose': 'engineering_feasibility_not_heldout', 'versions': versions,
        'sources': hashes, 'cache_sha256': fast_geometry.CACHE_SHA256,
        'targets_ml': args.targets, 'seed': args.seed, 'randomized': args.randomize,
        'policy_parameters': policy.params, 'deadline_s': 5.0, 'post_observation_s': 2.0,
        'observation_end_physical_time_s': 7.0, 'kinematic_samples_per_physical_frame': 4,
        'physical_dt_s': 1/60, 'pressure_iterations': 6, 'minimum_substeps': 4,
        'cuda_graph': False, 'liquid_solver_modified': False}
    shared.write_json(args.output / 'protocol.json', declaration)
    env = fast_training_env.FastTrainingEnv(geometry=fast_geometry, control=fast_control)
    results = []
    start = time.perf_counter()
    for target in args.targets:
        def progress(event):
            shared.write_json(args.output / 'progress.json', {'state': 'running',
                'completed': len(results), 'wall_seconds': time.perf_counter() - start, **event})
        result = env.run_episode(target, args.seed, policy=policy, randomize=args.randomize,
            record_path=args.output / f'target_{target}.steps.jsonl.gz', progress=progress)
        extra = {'motion_metrics': dict(env.fast_metrics), 'controller': dict(env.last_controller.status)}
        # Invalid motion/numeric failures are not advanced further.
        try:
            extra['post_observation'] = (env.post_observe(args.output / f'target_{target}.post.jsonl.gz')
                                        if result['valid_physics'] else None)
        except Exception as error:
            extra['post_observation'] = {'passed': False, 'exception': type(error).__name__ + ': ' + str(error)}
        results.append({'result': result, **extra})
        shared.write_json(args.output / 'results.json', {'complete': False, 'results': results})
        print('FAST PILOT ' + json.dumps({'target': target, 'received': result['received_ml'],
            'error_ml': result['error_ml'], 'time_s': result['physical_time_s'],
            'complete': result['complete'], 'valid': result['valid_physics'],
            'error': result['error'], 'motion': extra['motion_metrics']}), flush=True)
    for relative, expected in hashes.items():
        if shared._sha(ROOT / relative) != expected:
            raise RuntimeError('A pilot source changed while the process ran.')
    shared.write_json(args.output / 'results.json', {'complete': True, 'results': results,
        'purpose': 'engineering_feasibility_not_heldout', 'wall_seconds': time.perf_counter() - start})
    shared.write_json(args.output / 'progress.json', {'state': 'complete', 'completed': len(results)})


if __name__ == '__main__':
    main()
