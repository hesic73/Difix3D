#!/usr/bin/env python3
"""
Multi-GPU video processing script using Difix pipeline.
Processes videos from <dataset_dir>/<scene_id:05>.mp4 format.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing import Manager, Process, Queue
from pathlib import Path
from typing import Annotated, List

import imageio
import numpy as np
import torch
import tyro
from PIL import Image
from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from pipeline_difix import DifixPipeline

console = Console()


@dataclass
class WorkerArgs:
    """Arguments passed to worker processes."""
    model_id: str
    prompt: str
    num_inference_steps: int
    timesteps: List[int] | None
    guidance_scale: float
    fps: int


def parse_ranges(ranges_str: str | None) -> List[tuple[int | None, int | None]]:
    """
    Parse range specification string into list of (start, end) tuples.

    Args:
        ranges_str: String like "100:200,250:-1" where -1 means open-ended

    Returns:
        List of (start, end) tuples. end=None means open-ended.

    Examples:
        "100:200" -> [(100, 200)]
        "100:200,250:-1" -> [(100, 200), (250, None)]
        "0:50,100:150,200:-1" -> [(0, 50), (100, 150), (200, None)]
    """
    if not ranges_str:
        return [(None, None)]

    ranges = []
    for range_part in ranges_str.split(','):
        range_part = range_part.strip()
        if ':' not in range_part:
            raise ValueError(f"Invalid range format: {range_part}. Expected 'start:end'")

        start_str, end_str = range_part.split(':', 1)
        start = int(start_str.strip()) if start_str.strip() else None
        end_val = end_str.strip()

        if end_val == '-1' or end_val == '':
            end = None
        else:
            end = int(end_val)

        ranges.append((start, end))

    return ranges


def in_ranges(scene_id: int, ranges: List[tuple[int | None, int | None]]) -> bool:
    """
    Check if a scene_id falls within any of the specified ranges.

    Args:
        scene_id: Scene ID to check
        ranges: List of (start, end) tuples

    Returns:
        True if scene_id is in any range, False otherwise
    """
    if ranges == [(None, None)]:
        return True

    for start, end in ranges:
        in_range = True

        # Check lower bound
        if start is not None and scene_id < start:
            in_range = False

        # Check upper bound
        if end is not None and scene_id >= end:
            in_range = False

        # If scene_id is in this range, return True
        if in_range:
            return True

    return False


def find_pending_videos(
    dataset_dir: Path,
    output_dir: Path,
    ranges: List[tuple[int | None, int | None]] | None = None,
    skip_existing: bool = True,
    descending: bool = False,
) -> List[Path]:
    """
    Find videos that need processing based on range and existence filters.

    Args:
        dataset_dir: Directory containing input videos
        output_dir: Output directory for checking existing files
        ranges: List of (start, end) tuples for filtering
        skip_existing: If True, skip videos that already exist in output_dir
        descending: If True, process videos in descending order (default: ascending)

    Returns:
        List of video file paths to process
    """
    if ranges is None:
        ranges = [(None, None)]

    all_videos = list(dataset_dir.glob("*.mp4"))

    # Build list of (scene_id, video_path) tuples for proper numeric sorting
    video_tuples = []
    for video_path in all_videos:
        try:
            scene_id = int(video_path.stem)
            video_tuples.append((scene_id, video_path))
        except ValueError:
            # Skip files that don't match expected naming format
            continue

    # Sort by scene_id (numeric), not by filename (string)
    video_tuples.sort(key=lambda x: x[0], reverse=descending)

    # Filter by range and existence
    pending = []
    for scene_id, video_path in video_tuples:
        # Check if in range
        if not in_ranges(scene_id, ranges):
            continue

        # Check if already exists
        if skip_existing:
            output_path = output_dir / video_path.name
            if output_path.exists():
                continue

        pending.append(video_path)

    return pending


def ensure_even_dimensions(arr: np.ndarray) -> np.ndarray:
    """
    Ensure height and width are even numbers for H.264 encoding compatibility.
    Crops if necessary to avoid interpolation artifacts.
    """
    h, w = arr.shape[:2]
    new_h = h if h % 2 == 0 else h - 1
    new_w = w if w % 2 == 0 else w - 1
    if new_h != h or new_w != w:
        arr = arr[:new_h, :new_w]
    return arr


def parse_timesteps(value: str | None) -> List[int] | None:
    """Parse comma-separated timestep values."""
    if value is None or not value.strip():
        return None
    text = value.strip()
    if text.lower() in {"none", "null"}:
        return None
    pieces = [item.strip() for item in text.split(",") if item.strip()]
    return [int(piece) for piece in pieces] if pieces else None


def process_video(
    video_path: Path,
    output_path: Path,
    pipe: DifixPipeline,
    worker_args: WorkerArgs,
) -> int:
    """
    Process a single video file through the Difix pipeline.
    Returns the number of frames processed.
    """
    # Read input video
    reader = imageio.get_reader(str(video_path))
    fps = reader.get_meta_data().get('fps', worker_args.fps)

    frames_processed = 0
    with imageio.get_writer(str(output_path), fps=fps) as writer:
        for frame in reader:
            # Convert frame to PIL Image format
            input_image = Image.fromarray(frame)

            # Prepare pipeline arguments
            pipe_kwargs = {
                "image": input_image,
                "num_inference_steps": worker_args.num_inference_steps,
                "guidance_scale": worker_args.guidance_scale,
            }

            if worker_args.timesteps:
                pipe_kwargs["timesteps"] = worker_args.timesteps

            # Run inference
            result = pipe(worker_args.prompt, **pipe_kwargs).images[0]

            # Write output frame
            frame_array = ensure_even_dimensions(np.array(result))
            writer.append_data(frame_array)
            frames_processed += 1

    reader.close()
    return frames_processed


def worker_process(
    gpu_id: int,
    task_queue: Queue,
    status_dict: dict,
    dataset_dir: Path,
    output_dir: Path,
    worker_args: WorkerArgs,
):
    """
    Worker process that processes videos on a specific GPU.

    Args:
        gpu_id: GPU device ID
        task_queue: Queue of video paths to process
        status_dict: Shared dict for status reporting
        dataset_dir: Root data directory
        output_dir: Output directory for videos
        worker_args: Worker arguments (model config, etc.)
    """
    # Set GPU for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda:0')  # Always 0 since CUDA_VISIBLE_DEVICES is set

    # Random delay to avoid filesystem conflicts
    time.sleep(random.uniform(0, 1.0))

    # Initialize status for this GPU
    status_dict[gpu_id] = {
        'status': 'initializing',
        'video': None,
        'completed': 0,
        'failed': 0,
        'total_frames': 0,
        'start_time': None
    }

    try:
        # Load model
        pipe = DifixPipeline.from_pretrained(
            worker_args.model_id,
            trust_remote_code=True,
            local_files_only=False,
        )
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)

        # Update status to idle
        status_dict[gpu_id] = {
            'status': 'idle',
            'video': None,
            'completed': 0,
            'failed': 0,
            'total_frames': 0,
            'start_time': None
        }

        # Process videos from queue
        while True:
            try:
                video_path = task_queue.get(timeout=1)
            except:
                if task_queue.empty():
                    # Mark as finished
                    current = status_dict[gpu_id]
                    status_dict[gpu_id] = {
                        'status': 'finished',
                        'video': None,
                        'completed': current['completed'],
                        'failed': current['failed'],
                        'total_frames': current['total_frames'],
                        'start_time': None
                    }
                    break
                continue

            if video_path is None:  # Poison pill
                current = status_dict[gpu_id]
                status_dict[gpu_id] = {
                    'status': 'finished',
                    'video': None,
                    'completed': current['completed'],
                    'failed': current['failed'],
                    'total_frames': current['total_frames'],
                    'start_time': None
                }
                break

            # Update status to processing
            current = status_dict[gpu_id]
            status_dict[gpu_id] = {
                'status': 'processing',
                'video': video_path.name,
                'completed': current['completed'],
                'failed': current['failed'],
                'total_frames': current['total_frames'],
                'start_time': time.time()
            }

            try:
                output_path = output_dir / video_path.name
                frames_processed = process_video(video_path, output_path, pipe, worker_args)

                # Update status - success
                current = status_dict[gpu_id]
                status_dict[gpu_id] = {
                    'status': 'idle',
                    'video': None,
                    'completed': current['completed'] + 1,
                    'failed': current['failed'],
                    'total_frames': current['total_frames'] + frames_processed,
                    'start_time': None
                }

                # Force cleanup
                import gc
                torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                # Update status - failed
                print(f"[GPU {gpu_id}] Error processing video: {e}")
                import traceback
                traceback.print_exc()
                current = status_dict[gpu_id]
                status_dict[gpu_id] = {
                    'status': 'idle',
                    'video': None,
                    'completed': current['completed'],
                    'failed': current['failed'] + 1,
                    'total_frames': current['total_frames'],
                    'start_time': None
                }

    except Exception as e:
        # Model loading or other critical error
        print(f"[GPU {gpu_id}] CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        status_dict[gpu_id] = {
            'status': 'error',
            'video': None,
            'completed': status_dict.get(gpu_id, {}).get('completed', 0),
            'failed': status_dict.get(gpu_id, {}).get('failed', 0),
            'total_frames': status_dict.get(gpu_id, {}).get('total_frames', 0),
            'start_time': None
        }


def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds is None:
        return "N/A"
    return str(timedelta(seconds=int(seconds)))


def create_status_display(status_dict, total_videos, start_time, gpu_ids):
    """Create a rich layout for displaying multi-GPU status."""

    # Calculate overall statistics
    total_completed = sum(status_dict.get(gpu_id, {}).get('completed', 0) for gpu_id in gpu_ids)
    total_failed = sum(status_dict.get(gpu_id, {}).get('failed', 0) for gpu_id in gpu_ids)
    total_frames = sum(status_dict.get(gpu_id, {}).get('total_frames', 0) for gpu_id in gpu_ids)
    total_processed = total_completed + total_failed

    elapsed_time = time.time() - start_time

    # Calculate ETA
    if total_processed > 0:
        avg_time = elapsed_time / total_processed
        remaining = total_videos - total_processed
        eta = avg_time * remaining
    else:
        eta = None

    # Create summary panel
    summary_text = Text()
    summary_text.append("Progress: ", style="bold")
    summary_text.append(f"{total_processed}/{total_videos} ", style="bold cyan")
    summary_text.append("(", style="dim")
    summary_text.append(f"✓ {total_completed} ", style="bold green")
    summary_text.append(f"✗ {total_failed}", style="bold red")
    summary_text.append(")", style="dim")

    summary_text.append(" │ ", style="dim")
    summary_text.append(f"Frames: {total_frames} ", style="bold yellow")

    summary_text.append(" │ ", style="dim")
    summary_text.append("Elapsed: ", style="bold")
    summary_text.append(format_time(elapsed_time), style="yellow")

    if eta is not None:
        summary_text.append(" │ ", style="dim")
        summary_text.append("ETA: ", style="bold")
        summary_text.append(format_time(eta), style="yellow")

    summary_panel = Panel(
        summary_text,
        title="[bold magenta]Overall Progress[/bold magenta]",
        border_style="cyan",
        box=box.ROUNDED
    )

    # Create GPU status table
    gpu_table = Table(
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED,
        padding=(0, 1)
    )
    gpu_table.add_column("GPU", style="cyan", width=6)
    gpu_table.add_column("Status", width=12)
    gpu_table.add_column("Current Video", width=20)
    gpu_table.add_column("Completed", justify="right", width=10)
    gpu_table.add_column("Failed", justify="right", width=8)
    gpu_table.add_column("Frames", justify="right", width=10)
    gpu_table.add_column("Processing Time", justify="right", width=16)

    for gpu_id in gpu_ids:
        info = status_dict.get(gpu_id, {})
        status = info.get('status', 'unknown')
        video = info.get('video')
        completed = info.get('completed', 0)
        failed = info.get('failed', 0)
        frames = info.get('total_frames', 0)
        start_time_gpu = info.get('start_time')

        # Format status with colors
        if status == 'processing':
            status_str = "[bold yellow]Processing[/bold yellow]"
        elif status == 'idle':
            status_str = "[green]Idle[/green]"
        elif status == 'initializing':
            status_str = "[blue]Loading...[/blue]"
        elif status == 'finished':
            status_str = "[bold green]Finished[/bold green]"
        elif status == 'error':
            status_str = "[bold red]Error[/bold red]"
        else:
            status_str = "[dim]Unknown[/dim]"

        # Format video name
        video_str = video if video is not None else "-"
        if video_str != "-" and len(video_str) > 20:
            video_str = video_str[:17] + "..."

        # Format processing time
        if start_time_gpu is not None:
            proc_time = time.time() - start_time_gpu
            time_str = f"{proc_time:.1f}s"
        else:
            time_str = "-"

        gpu_table.add_row(
            f"GPU {gpu_id}",
            status_str,
            video_str,
            f"[green]{completed}[/green]",
            f"[red]{failed}[/red]" if failed > 0 else "0",
            f"[yellow]{frames}[/yellow]",
            time_str
        )

    gpu_panel = Panel(
        gpu_table,
        title="[bold magenta]GPU Status[/bold magenta]",
        border_style="cyan",
        box=box.ROUNDED
    )

    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(summary_panel, size=3),
        Layout(gpu_panel)
    )

    return layout


def main(
    dataset_dir: Annotated[Path, tyro.conf.Positional],
    output_dir: Annotated[Path, tyro.conf.Positional],
    ranges: Annotated[str | None, tyro.conf.arg(help="Comma-separated ranges (e.g., '100:200,250:-1'). Use -1 for open-ended.")] = None,
    overwrite: Annotated[bool, tyro.conf.arg(help="Overwrite existing output videos. By default, existing videos are skipped.")] = False,
    descending: Annotated[bool, tyro.conf.arg(help="Process videos in descending order by ID. By default, processes in ascending order.")] = False,
    no_tui: Annotated[bool, tyro.conf.arg(help="Disable TUI live display for debugging. Shows plain text output instead.")] = False,
    gpus: Annotated[str | None, tyro.conf.arg(help="Comma-separated GPU IDs (e.g., '0,1,2,3'). If not specified, uses all available GPUs.")] = None,
    model_id: Annotated[str, tyro.conf.arg(help="Difix model ID or local path")] = "nvidia/difix",
    prompt: Annotated[str, tyro.conf.arg(help="Inference prompt")] = "remove degradation",
    num_inference_steps: Annotated[int, tyro.conf.arg(help="Number of diffusion inference steps")] = 1,
    timesteps: Annotated[str, tyro.conf.arg(help="Custom timesteps (comma-separated). Use 'none' to disable.")] = "199",
    guidance_scale: Annotated[float, tyro.conf.arg(help="Guidance scale for inference")] = 0.0,
    fps: Annotated[int, tyro.conf.arg(help="Default output FPS if not detected from input video")] = 24,
):
    """Multi-GPU video processing using Difix pipeline."""

    # Validate paths
    dataset_dir = dataset_dir.expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse timesteps and ranges
    parsed_timesteps = parse_timesteps(timesteps)
    parsed_ranges = parse_ranges(ranges) if ranges else None

    # Create worker args
    worker_args = WorkerArgs(
        model_id=model_id,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        timesteps=parsed_timesteps,
        guidance_scale=guidance_scale,
        fps=fps,
    )

    # Determine GPU IDs
    if gpus:
        gpu_ids = [int(x.strip()) for x in gpus.split(',')]
    else:
        # Use all available GPUs
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            console.print("[red]No GPUs available![/red]")
            return
        gpu_ids = list(range(gpu_count))

    # Print header
    console.print("\n[bold magenta]╔═══════════════════════════════════════════════════════════════╗[/bold magenta]")
    console.print("[bold magenta]║[/bold magenta]        [bold cyan]Difix Multi-GPU Video Processing Pipeline[/bold cyan]        [bold magenta]║[/bold magenta]")
    console.print("[bold magenta]╚═══════════════════════════════════════════════════════════════╝[/bold magenta]\n")

    # Find pending videos
    console.print("[bold cyan]Scanning for videos to process...[/bold cyan]")
    skip_existing = not overwrite  # By default skip existing, unless --overwrite is used
    pending_videos = find_pending_videos(
        dataset_dir, output_dir, ranges=parsed_ranges, skip_existing=skip_existing, descending=descending
    )

    if not pending_videos:
        console.print(f"[green]No videos to process. All videos already exist![/green]")
        return

    # Print info
    console.print(f"[bold]Found {len(pending_videos)} videos to process[/bold]")
    if parsed_ranges and parsed_ranges != [(None, None)]:
        ranges_str = ", ".join([f"[{s if s is not None else '0'}:{e if e is not None else '∞'})" for s, e in parsed_ranges])
        console.print(f"Range filter: {ranges_str}")
    if overwrite:
        console.print(f"Overwrite mode: [yellow]enabled (will reprocess existing videos)[/yellow]")
    else:
        console.print(f"Skip existing: [cyan]enabled (default)[/cyan]")

    # Show processing order
    order_str = "descending" if descending else "ascending"
    console.print(f"Processing order: [cyan]{order_str}[/cyan]")

    console.print(f"Dataset directory: [cyan]{dataset_dir}[/cyan]")
    console.print(f"Output directory: [cyan]{output_dir}[/cyan]")
    console.print(f"Using GPUs: [cyan]{', '.join(map(str, gpu_ids))}[/cyan]\n")

    # Create shared structures
    manager = Manager()
    task_queue = manager.Queue()
    status_dict = manager.dict()

    # Populate task queue
    console.print(f"[dim]Populating queue with {len(pending_videos)} videos...[/dim]")
    for video_path in pending_videos:
        task_queue.put(video_path)
    console.print(f"[dim]Queue populated. Adding {len(gpu_ids)} poison pills...[/dim]")

    # Add poison pills
    for _ in gpu_ids:
        task_queue.put(None)
    console.print(f"[dim]Queue ready with {len(pending_videos)} videos + {len(gpu_ids)} poison pills[/dim]")

    # Start worker processes
    console.print(f"[bold cyan]Starting {len(gpu_ids)} worker processes...[/bold cyan]\n")
    workers = []
    for gpu_id in gpu_ids:
        p = Process(
            target=worker_process,
            args=(gpu_id, task_queue, status_dict, dataset_dir, output_dir, worker_args)
        )
        p.start()
        workers.append(p)

    # Monitor progress with live display
    start_time = time.time()

    if no_tui:
        # Plain text mode for debugging
        console.print("[yellow]Monitoring workers (plain text mode)...[/yellow]\n")
        while any(p.is_alive() for p in workers):
            # Print status every 5 seconds
            time.sleep(5)
            print(f"\n[{time.strftime('%H:%M:%S')}] Worker status:")
            for gpu_id in gpu_ids:
                info = status_dict.get(gpu_id, {})
                status = info.get('status', 'unknown')
                video = info.get('video', '-')
                completed = info.get('completed', 0)
                failed = info.get('failed', 0)
                print(f"  GPU {gpu_id}: {status:12s} | Video: {video:20s} | Completed: {completed:3d} | Failed: {failed:3d}")
    else:
        # TUI mode
        with Live(
            create_status_display(status_dict, len(pending_videos), start_time, gpu_ids),
            refresh_per_second=2,
            console=console
        ) as live:
            while any(p.is_alive() for p in workers):
                live.update(create_status_display(status_dict, len(pending_videos), start_time, gpu_ids))
                time.sleep(0.5)

    # Wait for all workers to complete
    for p in workers:
        p.join()

    # Final summary
    total_time = time.time() - start_time
    total_completed = sum(status_dict.get(gpu_id, {}).get('completed', 0) for gpu_id in gpu_ids)
    total_failed = sum(status_dict.get(gpu_id, {}).get('failed', 0) for gpu_id in gpu_ids)
    total_frames = sum(status_dict.get(gpu_id, {}).get('total_frames', 0) for gpu_id in gpu_ids)

    console.print("\n")
    summary_table = Table(
        title="[bold magenta]Final Summary[/bold magenta]",
        box=box.DOUBLE,
        show_header=True,
        header_style="bold magenta"
    )
    summary_table.add_column("Metric", style="cyan", no_wrap=True)
    summary_table.add_column("Value", justify="right", style="green")

    summary_table.add_row("Total videos", str(len(pending_videos)))
    summary_table.add_row("Successful", f"[green]{total_completed}[/green]")
    summary_table.add_row("Failed", f"[red]{total_failed}[/red]" if total_failed > 0 else "0")
    summary_table.add_row("Success rate", f"{total_completed/(total_completed+total_failed)*100:.1f}%" if (total_completed+total_failed) > 0 else "N/A")
    summary_table.add_row("Total frames", f"[yellow]{total_frames}[/yellow]")
    summary_table.add_row("", "")
    summary_table.add_row("GPUs used", str(len(gpu_ids)))
    summary_table.add_row("Total time", format_time(total_time))
    summary_table.add_row("Avg per video", f"{total_time/(total_completed+total_failed):.1f}s" if (total_completed+total_failed) > 0 else "N/A")

    console.print(summary_table)
    console.print(f"\n[bold green]✓ Multi-GPU batch processing completed![/bold green]")
    console.print(f"[cyan]Videos saved to: {output_dir}[/cyan]\n")


if __name__ == "__main__":
    # CUDA requires 'spawn' method for multiprocessing
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    tyro.cli(main)
