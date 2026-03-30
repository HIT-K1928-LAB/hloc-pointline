import argparse
import contextlib
import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pycolmap
from tqdm import tqdm

from . import logger
from .utils.database import COLMAPDatabase
from .utils.geometry import compute_epipolar_errors
from .utils.io import get_keypoints, get_matches
from .utils.parsers import parse_retrieval




db = COLMAPDatabase.connect(database_path)
db.create_tables()

for i, camera in reconstruction.cameras.items():
    db.add_camera(
        camera.model.value,
        camera.width,
        camera.height,
        camera.params,
        camera_id=i,
        prior_focal_length=True,
    )

for i, image in reconstruction.images.items():
    db.add_image(image.name, image.camera_id, image_id=i)

db.commit()
db.close()
return {image.name: i for i, image in reconstruction.images.items()}