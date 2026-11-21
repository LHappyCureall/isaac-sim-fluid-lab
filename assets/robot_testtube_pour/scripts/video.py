"""Deterministic cached-pose capture: exactly N native frames and a verified MP4."""

import asyncio
import inspect
import json
import math
import os
from pathlib import Path
from fractions import Fraction
import shutil
import subprocess
import tempfile
import time


def _encoding_tools():
    packages = Path(os.environ.get('LOCALAPPDATA', str(Path.home() / 'AppData' / 'Local'))) / 'Microsoft/WinGet/Packages'
    candidates = list(packages.glob('Gyan.FFmpeg_*/ffmpeg-*/bin/ffmpeg.exe')) if packages.is_dir() else []
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for ffmpeg in candidates:
        ffprobe = ffmpeg.with_name('ffprobe.exe')
        if ffprobe.is_file():
            return str(ffmpeg), str(ffprobe)
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        raise RuntimeError('FFmpeg and FFprobe are required to export a verified MP4.')
    return ffmpeg, ffprobe


def _owns(demo):
    if getattr(demo, '_closed', False):
        return False
    check = getattr(demo, '_owns_scene', None)
    try:
        return bool(check()) if callable(check) else True
    except (RuntimeError, ReferenceError):
        return False


def _ensure_current(demo):
    if not _owns(demo):
        raise asyncio.CancelledError('The pouring scene was replaced.')


async def _process(args):
    # File-backed output prevents pipe-buffer deadlocks while polling. Retain
    # the exact child handle so cancellation never leaves FFmpeg running.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        options = {'stdin': subprocess.DEVNULL, 'stdout': stdout, 'stderr': stderr}
        if os.name == 'nt':
            options['creationflags'] = subprocess.CREATE_NO_WINDOW
        process = subprocess.Popen(args, **options)
        try:
            while process.poll() is None:
                await asyncio.sleep(.05)
            stdout.seek(0)
            stderr.seek(0)
            output = stdout.read().decode('utf-8', errors='replace')
            errors = stderr.read().decode('utf-8', errors='replace')
            if process.returncode:
                raise RuntimeError(f'{Path(args[0]).name} failed: {errors[-3000:]}')
            return output
        except BaseException:
            if process.poll() is None:
                process.terminate()
                deadline = time.monotonic() + 2.0
                while process.poll() is None and time.monotonic() < deadline:
                    try:
                        await asyncio.sleep(.02)
                    except asyncio.CancelledError:
                        pass
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=2.0)
            raise


async def _drain_capture(task, frame_path, app):
    if task is None:
        return True
    # Native screenshot completion can precede the disk write. Keep its
    # protected task and private folder alive even after scene cancellation.
    for _ in range(60):
        if task.done():
            try:
                task.result()
            except (Exception, asyncio.CancelledError):
                pass
            if frame_path is not None and frame_path.is_file() and frame_path.stat().st_size > 0:
                return True
        try:
            await app.next_update_async()
        except asyncio.CancelledError:
            pass
        except RuntimeError:
            break
    if not task.done():
        task.cancel()
    return False


async def export_demo_mp4(demo, output_path, duration=10.0, fps=30, width=1280, height=720):
    """Capture cached source poses without timeline-driven simulation updates.

    Only the explicitly named MP4 and adjacent video_export.json are replaced.
    The temporary frame directory is private to this call. The timeline remains
    paused on completion or failure, with the original source pose restored.
    """
    import carb
    import omni.kit.app
    import omni.timeline
    from omni.kit.viewport.utility import get_active_viewport, next_viewport_frame_async, capture_viewport_to_file

    _ensure_current(demo)
    duration, fps = float(duration), float(fps)
    if not all(math.isfinite(x) and x > 0 for x in (duration, fps)):
        raise ValueError('Duration and FPS must be finite positive values.')
    width, height = int(width), int(height)
    if min(width, height) <= 0 or width % 2 or height % 2:
        raise ValueError('H.264 yuv420p export requires positive even width and height.')
    count = int(round(duration * fps))
    if count < 2:
        raise ValueError('The video must contain at least two frames.')
    output = Path(output_path).expanduser().resolve()
    if output.suffix.lower() != '.mp4':
        raise ValueError('The output path must end in .mp4.')
    if getattr(demo, 'busy', False):
        raise RuntimeError('The pouring demo is busy; wait for its current operation.')
    ffmpeg, ffprobe = _encoding_tools()
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError('An active Isaac Sim viewport is required.')
    source_end = float(demo.video_source_time(duration, duration))
    last_tilt = float(demo.geometry.mouth_pose(source_end)[1])
    if not math.isfinite(source_end) or source_end < 0 or abs(last_tilt) > 1e-6:
        raise ValueError('The requested video must finish with the test tube upright.')

    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    owned = Path(tempfile.mkdtemp(prefix='robot_pour_video_', dir=str(parent))).resolve()
    if owned.parent != parent or not owned.name.startswith('robot_pour_video_'):
        raise RuntimeError('Temporary video directory is outside the requested output folder.')
    timeline = omni.timeline.get_timeline_interface()
    settings = carb.settings.get_settings()
    old_time, old_busy = float(demo.fluid.time), bool(demo.busy)
    old_exporting = bool(getattr(demo, "_exporting", False))
    old_resolution = tuple(viewport.resolution)
    old_mode = settings.get('/rtx/rendermode')
    label = getattr(demo, 'status_label', None)
    old_label = label.text if label is not None else None
    started = time.perf_counter()
    pending_capture, pending_frame = None, None
    app = omni.kit.app.get_app()
    demo.busy = True
    demo._exporting = True
    timeline.pause()
    try:
        viewport.resolution = (width, height)
        settings.set_string('/rtx/rendermode', 'RealTimePathTracing')
        for k in range(count):
            _ensure_current(demo)
            video_time = duration * k / (count - 1)
            source_time = float(demo.video_source_time(video_time, duration))
            if not math.isfinite(source_time):
                raise ValueError(f'Invalid source time for video frame {k}.')
            result = demo.seek_recorded(source_time)
            if inspect.isawaitable(result):
                await result
            _ensure_current(demo)
            if label is not None and (k % 10 == 0 or k == count - 1):
                label.text = f'Exporting MP4: {k + 1}/{count} frames'
            await next_viewport_frame_async(viewport, 2)
            _ensure_current(demo)
            frame_path = owned / f'frame_{k:04d}.png'
            capture = capture_viewport_to_file(viewport, file_path=str(frame_path))
            pending_frame = frame_path
            pending_capture = asyncio.ensure_future(capture.wait_for_result())
            # FileCapture schedules its disk write after the render callback.
            # Keep the pose frozen while those completion frames are pumped.
            await asyncio.shield(pending_capture)
            _ensure_current(demo)
            for _ in range(60):
                if frame_path.is_file() and frame_path.stat().st_size > 0:
                    break
                await app.next_update_async()
                _ensure_current(demo)
            if not frame_path.is_file() or frame_path.stat().st_size == 0:
                raise RuntimeError(f'Native viewport capture did not save frame {k}.')
            pending_capture, pending_frame = None, None
        frames = list(owned.glob('frame_*.png'))
        if len(frames) != count:
            raise RuntimeError(f'Expected {count} captured frames, found {len(frames)}.')
        _ensure_current(demo)
        if label is not None:
            label.text = 'Encoding and verifying MP4...'
        encoded = owned / 'encoded.mp4'
        await _process([ffmpeg, '-hide_banner', '-y', '-framerate', str(fps), '-start_number', '0',
            '-i', str(owned / 'frame_%04d.png'), '-frames:v', str(count), '-an', '-c:v', 'libx264',
            '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(encoded)])
        _ensure_current(demo)
        probe = json.loads(await _process([ffprobe, '-v', 'error', '-count_frames', '-select_streams', 'v:0',
            '-show_entries', 'stream=nb_read_frames,nb_frames,r_frame_rate,avg_frame_rate,duration,width,height:format=duration',
            '-of', 'json', str(encoded)]))
        _ensure_current(demo)
        stream = probe['streams'][0]
        actual_count = int(stream.get('nb_read_frames', stream.get('nb_frames', -1)))
        actual_fps = float(Fraction(stream['avg_frame_rate']))
        actual_duration = float(probe['format']['duration'])
        expected_duration = count / fps
        if (actual_count != count or abs(actual_fps - fps) > 1e-3
                or abs(actual_duration - expected_duration) > .002
                or int(stream['width']) != width or int(stream['height']) != height):
            raise RuntimeError(f'MP4 verification failed: {probe}')
        report = {'success': True, 'output_path': str(output), 'frame_count': actual_count,
            'fps': actual_fps, 'duration_seconds': actual_duration,
            'requested_duration_seconds': duration, 'resolution': [width, height],
            'source_start_seconds': float(demo.video_source_time(0.0, duration)),
            'source_end_seconds': source_end, 'last_tilt_radians': last_tilt,
            'ends_upright': True, 'elapsed_seconds': time.perf_counter() - started,
            'render_mode': 'RealTimePathTracing', 'ffmpeg': ffmpeg, 'ffprobe': ffprobe,
            'verified_with_ffprobe': True, 'encoding': 'H.264/libx264 CRF18 yuv420p'}
        _ensure_current(demo)
        os.replace(encoded, output)
        (parent / 'video_export.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        return report
    finally:
        # A replaced demo must never pause or restore the NEW scene. Draining
        # an already scheduled capture only pumps completion; it writes no USD.
        capture_finished = await _drain_capture(pending_capture, pending_frame, app)
        try:
            if _owns(demo):
                timeline.pause()
                result = demo.seek_recorded(old_time)
                if inspect.isawaitable(result):
                    await result
        finally:
            if _owns(demo):
                viewport.resolution = old_resolution
                if old_mode is not None:
                    settings.set_string('/rtx/rendermode', old_mode)
                if label is not None:
                    label.text = old_label
            demo.busy = old_busy
            demo._exporting = old_exporting
            # Never remove a directory while a native capture may still write.
            # If shutdown prevented bounded draining, retain this private folder.
            resolved = owned.resolve()
            if (capture_finished and resolved.parent == parent
                    and resolved.name.startswith('robot_pour_video_')):
                shutil.rmtree(resolved)
