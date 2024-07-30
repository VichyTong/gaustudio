import sys
import argparse
import os
import time
import logging
from datetime import datetime
import torch
import json
from pathlib import Path
import cv2
import torchvision
from tqdm import tqdm
import vdbfusion
import trimesh
import numpy as np
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import base64
from gaustudio.utils.cameras_utils import JSON_to_camera

app = Flask(__name__)
CORS(app)


def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to config file", default="vanilla")
    parser.add_argument("--gpu", default="0", help="GPU(s) to be used")
    parser.add_argument("--sh", default=0, type=int, help="default SH degree")
    args, extras = parser.parse_known_args()

    # set CUDA_VISIBLE_DEVICES then import pytorch-lightning
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    n_gpus = len(args.gpu.split(","))

    from gaustudio.utils.misc import load_config
    from gaustudio import models, datasets, renderers

    # parse YAML config to OmegaConf
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, "../configs", args.config + ".yaml")
    config = load_config(config_path, cli_args=extras)
    config.cmd_args = vars(args)

    global pcd, renderer
    pcd = models.make(config.model.pointcloud)
    renderer = renderers.make(config.renderer)
    pcd.active_sh_degree = args.sh


def fit_plane(points):
    points = np.array(points)
    A = np.hstack((points, np.ones((points.shape[0], 1))))
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    plane_params = Vt[-1]
    plane_params /= np.linalg.norm(plane_params[:3])
    
    return plane_params

def project_to_plane(point, normal):
    normal = np.array(normal)
    point = np.array(point)
    d = np.dot(normal, point)
    projection = point - d * normal
    return projection

@app.route("/load_model", methods=["POST"])
def load_model():
    model_path = request.json.get("model_path")
    model_path = f"../../data/{model_path}"
    global work_dir

    if os.path.isdir(model_path):
        loaded_iter = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
        work_dir = os.path.join(
            model_path, "renders", "iteration_{}".format(loaded_iter)
        )

        print("Loading trained model at iteration {}".format(loaded_iter))
        pcd.load(
            os.path.join(
                model_path,
                "point_cloud",
                "iteration_" + str(loaded_iter),
                "point_cloud.ply",
            )
        )
    elif model_path.endswith(".ply"):
        work_dir = os.path.join(
            os.path.dirname(model_path), os.path.basename(model_path)[:-4]
        )
        pcd.load(model_path)
    else:
        print("Model not found at {}".format(model_path))
    pcd.to("cuda")

    global center
    center = pcd.calculate_center()

    global camera_json
    camera_path = os.path.join(model_path, "cameras.json")
    with open(camera_path, "r") as f:
        camera_data = json.load(f)

    points_list = []
    for camera in camera_data:
        points_list.append(camera["position"])
    plane = fit_plane(points_list)
    global axis_z
    axis_z = plane[:3]

    first_frame = camera_data[0]["position"]
    projected_point = project_to_plane(first_frame, axis_z)

    global axis_x
    axis_x = projected_point / np.linalg.norm(projected_point)
    global axis_y
    axis_y = np.cross(axis_z, axis_x)

    camera_json = camera_data[0]

    return jsonify({"message": f"Model loaded from {model_path}"})


@app.route("/adjust_f", methods=["POST"])
def adjust_f():
    delta_f = request.json.get("delta_f")
    camera_json["fx"] += delta_f
    camera_json["fy"] += delta_f

    return jsonify({"message": f"Adjusted focal length by {delta_f}"})


@app.route("/adjust_position", methods=["POST"])
def adjust_position():
    delta_x = request.json.get("delta_x")
    delta_y = request.json.get("delta_y")
    delta_z = request.json.get("delta_z")

    camera_json["position"][0] -= delta_x * axis_x[0] + delta_y * axis_y[0] + delta_z * axis_z[0]
    camera_json["position"][1] -= delta_x * axis_x[1] + delta_y * axis_y[1] + delta_z * axis_z[1]
    camera_json["position"][2] -= delta_x * axis_x[2] + delta_y * axis_y[2] + delta_z * axis_z[2]

    return jsonify(
        {
            "message": f"Adjusted the position of object by ({delta_x}, {delta_y}, {delta_z})"
        }
    )


@app.route("/adjust_rotation", methods=["POST"])
def adjust_rotation():
    alpha = request.json.get("alpha")
    position = np.array(camera_json['position'])
    current_rotation_matrix = np.array(camera_json["rotation"])
    object_center = np.array(center)

    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    I = np.eye(3)
    K = np.array([
        [0, -axis_z[2], axis_z[1]],
        [axis_z[2], 0, -axis_z[0]],
        [-axis_z[1], axis_z[0], 0]
    ])
    rotation_matrix = I + sin_alpha * K + (1 - cos_alpha) * K @ K

    translated_position = position - object_center
    rotated_position = rotation_matrix @ translated_position
    new_position = rotated_position + object_center
    
    new_rotation_matrix = rotation_matrix @ current_rotation_matrix

    camera_json['position'] = new_position.tolist()
    camera_json["rotation"] = new_rotation_matrix.tolist()

    return jsonify({"message": f"Adjusted the rotation of object by {alpha}"})


@app.route("/render", methods=["POST"])
def render():
    camera = JSON_to_camera(camera_json, "cuda")
    resolution = request.json.get("resolution", 1)
    white_background = request.json.get("white_background", False)
    return_base64 = request.json.get("return_base64", True)

    bg_color = [1, 1, 1] if white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_path = os.path.join(work_dir, "images")
    mask_path = os.path.join(work_dir, "masks")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)

    camera.downsample_scale(resolution)
    camera = camera.to("cuda")
    with torch.no_grad():
        render_pkg = renderer.render(camera, pcd)
    rendering = render_pkg["render"]
    invalid_mask = render_pkg["rendered_final_opacity"][0] < 0.5
    rendering[:, invalid_mask] = 0.0

    torchvision.utils.save_image(
        rendering, os.path.join(render_path, f"{camera.image_name}.png")
    )
    torchvision.utils.save_image(
        (~invalid_mask).float(), os.path.join(mask_path, f"{camera.image_name}.png")
    )

    image_path = os.path.join(render_path, f"{camera.image_name}.png")
    if return_base64:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return jsonify({"image": encoded_string})
    else:
        return send_file(image_path, mimetype="image/png")


if __name__ == "__main__":
    init()
    app.run(host="0.0.0.0", port=10024, debug=True)
