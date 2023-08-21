from typing import Tuple, Optional

import numpy as np

def poisson(img_shape: Tuple[int, int],
            accel: float,
            calib: Tuple[int, int] = (0, 0),
            dtype: np.dtype = np.complex128,
            crop_corner: bool = True,
            max_attempts: int = 30,
            tol: float = 0.1,
            seed: Optional[int] = None) -> np.ndarray:
  """Generate variable-density Poisson-disc sampling pattern.

  The function generates a variable density Poisson-disc sampling
  mask with density proportional to :math:`1 / (1 + s |r|)`,
  where :math:`r` represents the k-space radius, and :math:`s`
  represents the slope. A binary search is performed on the slope :math:`s`
  such that the resulting acceleration factor is close to the
  prescribed acceleration factor `accel`. The parameter `tol`
  determines how much they can deviate.

  Args:
    img_shape: Image shape, i.e., (height, width).
    accel: Target acceleration factor. Must be greater than 1.
    calib: Calibration shape.
    dtype: Data type.
    crop_corner: Whether to crop sampling corners.
    max_attempts: Maximum number of samples to reject in Poisson disc
      calculation.
    tol (float): Tolerance for how much the resulting acceleration can
      deviate form `accel`.
    seed (int): Random seed for initializing the NumPy random state.

  Returns:
    Poisson-disc sampling mask.

  References:
    Bridson, Robert. "Fast Poisson disk sampling in arbitrary dimensions."
    SIGGRAPH sketches. 2007.
  """
  if accel <= 1:
    raise ValueError(f'accel must be greater than 1, got {accel}')

  random_state = np.random.RandomState(seed)

  ny, nx = img_shape
  y, x = np.mgrid[:ny, :nx]
  x = np.maximum(abs(x - img_shape[-1] / 2) - calib[-1] / 2, 0)
  x /= x.max()
  y = np.maximum(abs(y - img_shape[-2] / 2) - calib[-2] / 2, 0)
  y /= y.max()
  r = np.sqrt(x**2 + y**2)

  slope_max = max(nx, ny)
  slope_min = 0
  while slope_min < slope_max:
    slope = (slope_max + slope_min) / 2
    radius_x = np.clip((1 + r * slope) * nx / max(nx, ny), 1, None)
    radius_y = np.clip((1 + r * slope) * ny / max(nx, ny), 1, None)
    mask = _poisson(img_shape[-1], img_shape[-2], max_attempts, radius_x,
                    radius_y, calib, random_state)
    if crop_corner:
      mask *= r < 1

    actual_accel = img_shape[-1] * img_shape[-2] / np.sum(mask)

    if abs(actual_accel - accel) < tol:
      break
    if actual_accel < accel:
      slope_min = slope
    else:
      slope_max = slope

  if abs(actual_accel - accel) >= tol:
    raise ValueError(f'Cannot generate mask to satisfy accel={accel}.')

  mask = mask.reshape(img_shape).astype(dtype)
  return mask


def _poisson(nx, ny, max_attempts, radius_x, radius_y, calib, random_state):
  mask = np.zeros((ny, nx))

  # Add calibration region.
  mask[int(ny / 2 - calib[-2] / 2):int(ny / 2 + calib[-2] / 2),
       int(nx / 2 - calib[-1] / 2):int(nx / 2 + calib[-1] / 2)] = 1

  # Initialize active list.
  pxs = np.empty(nx * ny, np.int32)
  pys = np.empty(nx * ny, np.int32)
  pxs[0] = random_state.randint(0, nx)
  pys[0] = random_state.randint(0, ny)
  num_actives = 1
  while num_actives > 0:
    i = random_state.randint(0, num_actives)
    px = pxs[i]
    py = pys[i]
    rx = radius_x[py, px]
    ry = radius_y[py, px]

    # Attempt to generate point.
    done = False
    k = 0
    while not done and k < max_attempts:
      # Generate point randomly from r and 2 * r.
      v = (random_state.random() * 3 + 1)**0.5
      t = 2 * np.pi * random_state.random()
      qx = px + v * rx * np.cos(t)
      qy = py + v * ry * np.sin(t)

      # Reject if outside grid or close to other points.
      if qx >= 0 and qx < nx and qy >= 0 and qy < ny:
        startx = max(int(qx - rx), 0)
        endx = min(int(qx + rx + 1), nx)
        starty = max(int(qy - ry), 0)
        endy = min(int(qy + ry + 1), ny)

        done = True
        for x in range(startx, endx):
          for y in range(starty, endy):
            if (mask[y, x] == 1 and (((qx - x) / radius_x[y, x])**2 +
                                     ((qy - y) / (radius_y[y, x]))**2 < 1)):
              done = False
              break

      k += 1

    # Add point if done else remove from active list.
    if done:
      pxs[num_actives] = qx
      pys[num_actives] = qy
      mask[int(qy), int(qx)] = 1
      num_actives += 1
    else:
      pxs[i] = pxs[num_actives - 1]
      pys[i] = pys[num_actives - 1]
      num_actives -= 1

  return mask


def cartesian(img_shape, accel):
  """From https://github.com/yang-song/score_inverse_problems/blob/b56e3836b9d7e6a26d41d366203b8e56a6bb5d0b/cs.py"""
  # shape [Tuple]: (H, W)
  size = img_shape[0]
  n_keep = size / accel
  center_fraction = n_keep / 1000

  num_rows, num_cols = img_shape[0], img_shape[1]
  num_low_freqs = int(round(num_cols * center_fraction))

  # create the mask
  mask = np.zeros((num_rows, num_cols), dtype=np.float32)
  pad = (num_cols - num_low_freqs + 1) // 2
  mask[:, pad: pad + num_low_freqs] = True

  # determine acceleration rate by adjusting for the number of low frequencies
  adjusted_accel = (accel * (num_low_freqs - num_cols)) / (
      num_low_freqs * accel - num_cols
  )

  offset = round(adjusted_accel) // 2

  accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
  accel_samples = np.around(accel_samples).astype(np.uint32)
  mask[:, accel_samples] = True

  return mask