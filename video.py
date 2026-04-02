import numpy as np
import imageio.v3 as iio
from PIL import Image


def encode_video(frames: list[Image.Image], output_path: str, fps: int = 24) -> None:
    """Encode a list of PIL images to an MP4 video."""
    arrays = [np.array(f.convert("RGB")) for f in frames]
    iio.imwrite(
        output_path,
        arrays,
        fps=fps,
        codec="libx264",
        plugin="pyav",
    )
