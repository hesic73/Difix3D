"""
Single-GPU video processing script using Difix pipeline.
Processes videos from <dataset_dir>/<scene_id:05>.mp4 format.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated, List

import imageio
import numpy as np
import torch
import tyro
from PIL import Image
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from pipeline_difix import DifixPipeline

console = Console()


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


def gather_video_files(
    dataset_dir: Path,
    ranges: List[tuple[int | None, int | None]] | None = None,
    skip_existing: bool = True,
    output_dir: Path | None = None,
) -> List[Path]:
    """
    Collect all .mp4 files from dataset directory with optional filtering.
    Expects naming format: 00001.mp4, 00002.mp4, etc.

    Args:
        dataset_dir: Directory containing input videos
        ranges: Optional list of (start, end) tuples for filtering by scene ID
        skip_existing: If True, skip videos that already exist in output_dir
        output_dir: Output directory for checking existing files

    Returns:
        List of video file paths to process
    """
    if ranges is None:
        ranges = [(None, None)]

    all_videos = sorted(dataset_dir.glob("*.mp4"))
    if not all_videos:
        raise FileNotFoundError(f"No .mp4 files found in {dataset_dir}")

    filtered_videos = []
    for video_path in all_videos:
        # Extract scene_id from filename (e.g., "00001.mp4" -> 1)
        try:
            scene_id = int(video_path.stem)
        except ValueError:
            # Skip files that don't match the expected naming format
            continue

        # Check if in range
        if not in_ranges(scene_id, ranges):
            continue

        # Check if already exists
        if skip_existing and output_dir:
            output_path = output_dir / video_path.name
            if output_path.exists():
                continue

        filtered_videos.append(video_path)

    return filtered_videos


def process_video(
    video_path: Path,
    output_path: Path,
    pipe: DifixPipeline,
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    timesteps: List[int] | None,
    frame_progress: Progress,
    frame_task_id,
) -> int:
    """
    Process a single video file through the Difix pipeline.
    Returns the number of frames processed.
    """
    # Read input video
    reader = imageio.get_reader(str(video_path))
    fps = reader.get_meta_data().get('fps', 24)

    # Get total frame count
    total_frames = reader.count_frames()
    frame_progress.update(frame_task_id, total=total_frames)

    frames_processed = 0
    with imageio.get_writer(str(output_path), fps=fps) as writer:
        for frame in reader:
            # Convert frame to PIL Image format
            input_image = Image.fromarray(frame)

            # Prepare pipeline arguments
            pipe_kwargs = {
                "image": input_image,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
            }

            if timesteps:
                pipe_kwargs["timesteps"] = timesteps

            # Run inference
            result = pipe(prompt, **pipe_kwargs).images[0]

            # Write output frame
            frame_array = ensure_even_dimensions(np.array(result))
            writer.append_data(frame_array)
            frames_processed += 1

            # Update frame progress
            frame_progress.update(frame_task_id, advance=1)

    reader.close()
    return frames_processed


def parse_timesteps(value: str | None) -> List[int] | None:
    """Parse comma-separated timestep values."""
    if value is None or not value.strip():
        return None
    text = value.strip()
    if text.lower() in {"none", "null"}:
        return None
    pieces = [item.strip() for item in text.split(",") if item.strip()]
    return [int(piece) for piece in pieces] if pieces else None




def create_status_table(
    total_videos: int,
    completed: int,
    failed: int,
    current_video: str,
    total_frames: int,
    elapsed_time: float,
) -> Table:
    """Create a status table with overall statistics."""
    table = Table(title="Processing Status", show_header=False, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")

    # Video statistics
    remaining = total_videos - completed - failed
    table.add_row("Total Videos", f"{total_videos}")
    table.add_row("✓ Completed", f"[green]{completed}[/green]")
    table.add_row("✗ Failed", f"[red]{failed}[/red]")
    table.add_row("⏳ Remaining", f"[yellow]{remaining}[/yellow]")

    # Current processing
    if current_video:
        table.add_row("Current", f"[bold cyan]{current_video}[/bold cyan]")

    # Frame statistics
    table.add_row("Total Frames", f"{total_frames}")

    # Time statistics
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    table.add_row("Elapsed", f"[cyan]{elapsed_str}[/cyan]")

    # ETA calculation
    if completed > 0:
        avg_time = elapsed_time / completed
        eta_seconds = avg_time * remaining
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        table.add_row("ETA", f"[cyan]{eta_str}[/cyan]")

    return table


def main(
    dataset_dir: Annotated[Path, tyro.conf.Positional],
    output_dir: Annotated[Path, tyro.conf.Positional],
    ranges: Annotated[str | None, tyro.conf.arg(help="Comma-separated ranges (e.g., '100:200,250:-1'). Use -1 for open-ended.")] = None,
    overwrite: Annotated[bool, tyro.conf.arg(help="Overwrite existing output videos. By default, existing videos are skipped.")] = False,
    model_id: Annotated[str, tyro.conf.arg(help="Difix model ID or local path")] = "nvidia/difix",
    prompt: Annotated[str, tyro.conf.arg(help="Inference prompt")] = "remove degradation",
    num_inference_steps: Annotated[int, tyro.conf.arg(help="Number of diffusion inference steps")] = 1,
    timesteps: Annotated[str, tyro.conf.arg(help="Custom timesteps (comma-separated). Use 'none' to disable.")] = "199",
    guidance_scale: Annotated[float, tyro.conf.arg(help="Guidance scale for inference")] = 0.0,
    fps: Annotated[int, tyro.conf.arg(help="Default output FPS if not detected from input video")] = 24,
    device: Annotated[str, tyro.conf.arg(help="CUDA device to use (e.g., 'cuda:0')")] = "cuda:0",
):
    """Single-GPU video processing using Difix pipeline."""

    # Validate paths
    dataset_dir = dataset_dir.expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse timesteps and ranges
    parsed_timesteps = parse_timesteps(timesteps)
    parsed_ranges = parse_ranges(ranges) if ranges else None
    skip_existing = not overwrite  # By default skip existing, unless --overwrite is used

    # Gather input videos with filtering
    video_files = gather_video_files(
        dataset_dir,
        ranges=parsed_ranges,
        skip_existing=skip_existing,
        output_dir=output_dir
    )

    if not video_files:
        console.print("[green]No videos to process. All videos already exist![/green]")
        return

    # Print configuration
    console.print("\n[bold cyan]═" * 30)
    console.print("[bold cyan]Difix Video Processing Pipeline")
    console.print("[bold cyan]═" * 30)
    console.print(f"[cyan]Dataset:[/cyan] {dataset_dir}")
    console.print(f"[cyan]Output:[/cyan] {output_dir}")
    console.print(f"[cyan]Model:[/cyan] {model_id}")
    console.print(f"[cyan]Device:[/cyan] {device}")
    console.print(f"[cyan]Videos to process:[/cyan] {len(video_files)}")

    # Show range and skip info
    if parsed_ranges and parsed_ranges != [(None, None)]:
        ranges_str = ", ".join([f"[{s if s is not None else '0'}:{e if e is not None else '∞'})" for s, e in parsed_ranges])
        console.print(f"[cyan]Range filter:[/cyan] {ranges_str}")
    if overwrite:
        console.print(f"[yellow]Overwrite mode:[/yellow] enabled (will reprocess existing)")
    else:
        console.print(f"[cyan]Skip existing:[/cyan] enabled (default)")

    console.print("[bold cyan]═" * 30 + "\n")

    # Initialize pipeline
    console.print("[yellow]Loading model...[/yellow]")
    torch.cuda.set_device(device)
    pipe = DifixPipeline.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=False,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    console.print("[green]✓ Model loaded successfully[/green]\n")

    # Initialize statistics
    completed_count = 0
    failed_count = 0
    total_frames_processed = 0
    start_time = time.time()

    # Create progress bars
    video_progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console,
    )

    frame_progress = Progress(
        TextColumn("[bold cyan]  ├─ Frames:"),
        BarColumn(complete_style="cyan", finished_style="cyan"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeRemainingColumn(),
        console=console,
    )

    # Process videos with live display
    with Live(console=console, refresh_per_second=4) as live:
        video_task = video_progress.add_task("Processing Videos", total=len(video_files))
        frame_task = frame_progress.add_task("Frames", total=0)

        for video_idx, video_path in enumerate(video_files):
            output_path = output_dir / video_path.name

            # Update display
            current_status = create_status_table(
                total_videos=len(video_files),
                completed=completed_count,
                failed=failed_count,
                current_video=video_path.name,
                total_frames=total_frames_processed,
                elapsed_time=time.time() - start_time,
            )

            display_group = Table.grid()
            display_group.add_row(Panel(current_status, border_style="blue"))
            display_group.add_row(video_progress)
            display_group.add_row(frame_progress)
            live.update(display_group)

            # Reset frame progress for new video
            frame_progress.reset(frame_task, completed=0, total=0)

            try:
                frames_processed = process_video(
                    video_path, output_path, pipe, prompt, num_inference_steps,
                    guidance_scale, parsed_timesteps, frame_progress, frame_task
                )
                total_frames_processed += frames_processed
                completed_count += 1

            except Exception as exc:
                failed_count += 1
                console.print(f"[red]✗ ERROR processing {video_path.name}: {exc}[/red]")
                # Continue processing other videos instead of raising

            # Update video progress
            video_progress.update(video_task, advance=1)

    # Final summary
    elapsed_time = time.time() - start_time
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    summary = Text()
    summary.append("\n")
    summary.append("PROCESSING COMPLETE\n\n", style="bold green")
    summary.append(f"Total time:      {elapsed_str}\n", style="cyan")
    summary.append(f"Total videos:    {len(video_files)}\n", style="bold")
    summary.append(f"✓ Successful:    {completed_count}\n", style="bold green")
    summary.append(f"✗ Failed:        {failed_count}\n", style="bold red")
    summary.append(f"Total frames:    {total_frames_processed}\n", style="bold")
    summary.append(f"\nOutput directory: {args.output_dir}\n", style="cyan")

    border_style = "green" if failed_count == 0 else "yellow"
    console.print(Panel(summary, border_style=border_style, title="Summary"))


if __name__ == "__main__":
    tyro.cli(main)
