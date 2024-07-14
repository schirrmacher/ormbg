# Must be at the top:
#  Make sure that is the first thing you import, as otherwise the import of third-party packages installed in the blender environment will fail.
import blenderproc as bproc
from blenderproc.python.renderer import RendererUtility

import os
import random
import argparse


def main(
    scene: str,
    output_dir: str,
    width: int,
    height: int,
    iterations: int,
    samples: int,
    min_distance: float,
    max_distance: float,
    camera_mode: str,
    lights_count: int,
) -> None:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

    bproc.init()
    RendererUtility.render_init()
    RendererUtility.set_max_amount_of_samples(samples)

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

    bproc.camera.set_resolution(width, height)

    target_names = ["target"]
    targets = [
        obj
        for obj in objects
        if hasattr(obj, "get_name")
        and callable(getattr(obj, "get_name"))
        and any(target in obj.get_name() for target in target_names)
    ]

    lights = []
    for _ in range(lights_count):
        lights.append(create_random_light())

    for i in range(iterations):

        for light in lights:
            light.set_location(random_light_location())

        if camera_mode == "frontal":
            start_angle = -135
            end_angle = -45
        elif camera_mode == "back":
            start_angle = 45
            end_angle = 135
        elif camera_mode == "frontal_and_back":
            random_bool = bool(random.getrandbits(1))
            start_angle = -135 if random_bool else 45
            end_angle = -45 if random_bool else 135

        point_of_interest = bproc.object.compute_poi(targets)

        while True:
            location = bproc.sampler.shell(
                center=[0, 0, random.uniform(1, 2)],
                radius_min=min_distance,
                radius_max=max_distance,
                elevation_min=-20,
                elevation_max=20,
                azimuth_min=start_angle,
                azimuth_max=end_angle,
            )

            random_camera_movement = [
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                random.uniform(-1, 1),
            ]
            point_of_interest = point_of_interest + random_camera_movement

            rotation_matrix = bproc.camera.rotation_from_forward_vec(
                point_of_interest - location
            )
            cam2world_matrix = bproc.math.build_transformation_mat(
                location, rotation_matrix
            )

            # Pose has to be added before evaluating object visibility
            bproc.camera.add_camera_pose(cam2world_matrix)

            targets_visible = all(
                bproc.camera.is_point_inside_camera_frustum(t.get_origin())
                for t in targets
            )
            if targets_visible:
                break
            else:
                bproc.utility.reset_keyframes()
                continue

        bproc.renderer.set_output_format(enable_transparency=True)
        data = bproc.renderer.render()
        bproc.writer.write_hdf5(output_dir, data, append_to_existing_output=True)
        bproc.utility.reset_keyframes()
        print(i)


def create_random_light():
    light = bproc.types.Light()
    light.set_type("POINT")
    light_location = random_light_location()
    light.set_location(light_location)
    light.set_energy(random.uniform(300, 500))
    return light


def random_light_location():
    location = bproc.sampler.shell(
        center=[0, random.uniform(0, 2), 2],
        radius_min=3,
        radius_max=3.5,
        elevation_min=-30,
        elevation_max=30,
    )
    return location


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
        "-w",
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
        "-min",
        type=float,
        default=2.0,
        help="Minimum distance for the camera",
    )

    parser.add_argument(
        "--max_distance",
        "-max",
        type=float,
        default=4.5,
        help="Maximum distance for the camera",
    )

    parser.add_argument(
        "--camera_mode",
        "-c",
        type=str,
        default="frontal_and_back",
        choices=["frontal", "back", "frontal_and_back"],
        help="Mode of camera positioning: 'frontal', 'back', or 'frontal_and_back'",
    )

    parser.add_argument(
        "--lights",
        "-l",
        type=int,
        default=2,
        help="Number of light sources randomly placed",
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
        args.camera_mode,
        args.lights,
    )
