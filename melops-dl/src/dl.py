"""Minimal yt-dlp wrapper for extracting video info."""

from typing import Any

from yt_dlp import YoutubeDL


def download(url: str, params: Any) -> Any:
    """
    Download a single video and return metadata.

    Uses extract_info with download=True. File paths are captured via
    post_hooks callbacks passed in params (typically Rust closures).

    Args:
        url: Video URL supported by yt-dlp
        params: Dictionary of parameters for yt-dlp (must include post_hooks)

    Returns:
        sanitized_info_dict: JSON-serializable metadata (id, title, duration, etc.)
    """
    with YoutubeDL(params) as ydl:
        info = ydl.extract_info(url, download=True)
        sanitized = ydl.sanitize_info(info)

        if sanitized is None:
            sanitized = info

        return sanitized
