# Must be at the top:
#  Make sure that is the first thing you import, as otherwise the import of third-party packages installed in the blender environment will fail.
import blenderproc as bproc
from blenderproc.python.renderer import RendererUtility

import os
import random
import argparse
import numpy as np


def compute_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def normalize_value(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)


def main(
    scene: str,
    output_dir: str,
    width: int,
    height: int,
    iterations: int,
    samples: int,
    min_distance: float,
    max_distance: float,
) -> None:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

    bproc.init()
    RendererUtility.render_init()
    RendererUtility.set_max_amount_of_samples(samples)

    # Load the objects into the scene
    objects = bproc.loader.load_blend(
        scene,
        obj_types=[
            "mesh",
            "curve",
            "curves",
            "hair",
        ],
        data_blocks=[
            "curves",
            "hair_curves",
            "materials",
            "meshes",
            "objects",
            "textures",
        ],
    )

    # Define the camera intrinsics
    bproc.camera.set_resolution(width, height)

    target_names = ["target"]
    targets = [
        obj
        for obj in objects
        if hasattr(obj, "get_name")
        and callable(getattr(obj, "get_name"))
        and any(target in obj.get_name() for target in target_names)
    ]

    poi = bproc.object.compute_poi(targets) + [0, 0, -0.5]

    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location(
        bproc.sampler.shell(
            center=[1, 2, 3],
            radius_min=4,
            radius_max=7,
            elevation_min=15,
            elevation_max=70,
        )
    )
    light.set_energy(500)

    for _ in range(iterations):
        location = bproc.sampler.shell(
            center=[0, 0, random.uniform(1, 2)],
            radius_min=min_distance,
            radius_max=max_distance,
            elevation_min=-30,
            elevation_max=30,
        )

        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location)

        cam2world_matrix = bproc.math.build_transformation_mat(
            location, rotation_matrix
        )
        bproc.camera.add_camera_pose(cam2world_matrix)

    bproc.renderer.set_output_format(enable_transparency=True)
    data = bproc.renderer.render()
    bproc.writer.write_hdf5(output_dir, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scene",
        "-s",
        nargs="?",
        default="blender/man.blend",
        help="Path to the scene.obj file",
    )

    parser.add_argument(
        "--output_dir",
        "-o",
        nargs="?",
        default="hdf5",
        help="Path to where the final files will be saved",
    )

    parser.add_argument(
        "--width",
        "-wt",
        type=int,
        default=256,
        help="Width of the camera resolution",
    )

    parser.add_argument(
        "--height",
        "-ht",
        type=int,
        default=256,
        help="Height of the camera resolution",
    )

    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=10,
        help="Number of iterations for rendering",
    )

    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=100,
        help="Number of iterations for rendering",
    )

    parser.add_argument(
        "--min_distance",
        "-mind",
        type=float,
        default=2.0,
        help="Minimum distance for the camera",
    )

    parser.add_argument(
        "--max_distance",
        "-maxd",
        type=float,
        default=4.0,
        help="Maximum distance for the camera",
    )

    args = parser.parse_args()

    main(
        args.scene,
        args.output_dir,
        args.width,
        args.height,
        args.iterations,
        args.samples,
        args.min_distance,
        args.max_distance,
    )
