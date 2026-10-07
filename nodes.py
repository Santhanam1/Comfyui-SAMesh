import torch
import os
import sys

# Strip EGL environment variables on Windows so PyOpenGL and pyrender use native WGL
if sys.platform == "win32" or os.name == "nt":
    os.environ.pop("PYOPENGL_PLATFORM", None)
    os.environ.pop("EGL_DEVICE_ID", None)

import requests
from pathlib import Path
import trimesh
import copy
import json
import numpy as np
import subprocess
from PIL import Image
from omegaconf import OmegaConf, DictConfig
from huggingface_hub import hf_hub_download
import folder_paths

try:
    import pyrender
except ImportError:
    print("\033[93mWarning: pyrender not found. The SamMeshRenderer node will not work. \nPlease install it (pip install pyrender) and potentially required system libraries.\033[0m")
    pyrender = None

samesh_src_dir_to_check = os.path.normpath(os.path.join("samesh-main", "src"))
samesh_src_dir_found = None
for p in sys.path:
    try:
        normalized_p = os.path.normpath(p)
        if normalized_p.endswith(samesh_src_dir_to_check):
            samesh_src_dir_found = p
            print(f"Found samesh src dir in sys.path: {p}")
            break
    except Exception:
        pass

comfyui_samesh_node_dir = os.path.dirname(os.path.realpath(__file__))
WORKER_SCRIPT_PATH = os.path.join(comfyui_samesh_node_dir, "_run_segmentation_worker.py")

print(f"ComfyUI-Sam-Mesh Node Dir: {comfyui_samesh_node_dir}")

read_mesh = None
SamModelMesh = None
remove_texture = None
colormap_faces_mesh = None
if samesh_src_dir_found:
    try:
        from samesh.data.loaders import read_mesh, remove_texture
        from samesh.models.sam_mesh import SamModelMesh, colormap_faces_mesh
    except ImportError as e:
        print(f"\033[31mError importing samesh functions (even though path seemed present): {e}\033[0m")
        read_mesh = None
        SamModelMesh = None
        remove_texture = None
        colormap_faces_mesh = None
else:
    print("\033[31mError: samesh source directory not found in sys.path after check. Check __init__.py path addition and submodule status.\033[0m")

if read_mesh is None:
    def read_mesh(*args, **kwargs):
        raise ImportError("samesh.data.loaders.read_mesh could not be imported.")

output_dir = folder_paths.get_output_directory()
temp_dir = folder_paths.get_temp_directory()
sam_model_dir = os.path.join(folder_paths.models_dir, "sam")
DEFAULT_OUTPUT_DIR = os.path.join(output_dir, "samesh")
DEFAULT_CACHE_DIR = os.path.join(temp_dir, "samesh_cache")

os.makedirs(sam_model_dir, exist_ok=True)

SAM_MODELS = {
    "SAM2 Hiera Large": {
        "checkpoint_filename": "sam2_hiera_large.pt",
        "config_filename": "sam2_hiera_l.yaml",
        "repo_id": "facebook/sam2-hiera-large",
        "config_url": "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2/sam2_hiera_l.yaml"
    },
    "SAM2 Hiera Base+": {
        "checkpoint_filename": "sam2_hiera_base_plus.pt",
        "config_filename": "sam2_hiera_b+.yaml",
        "repo_id": "facebook/sam2-hiera-base-plus",
        "config_url": "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2/sam2_hiera_b%2B.yaml"
    },
    "SAM2 Hiera Small": {
        "checkpoint_filename": "sam2_hiera_small.pt",
        "config_filename": "sam2_hiera_s.yaml",
        "repo_id": "facebook/sam2-hiera-small",
        "config_url": "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2/sam2_hiera_s.yaml"
    },
    "SAM2 Hiera Tiny": {
        "checkpoint_filename": "sam2_hiera_tiny.pt",
        "config_filename": "sam2_hiera_t.yaml",
        "repo_id": "facebook/sam2-hiera-tiny",
        "config_url": "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2/sam2_hiera_t.yaml"
    },
}
SAM_MODEL_NAMES = list(SAM_MODELS.keys())

class SamModelDownloader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (SAM_MODEL_NAMES, {"default": SAM_MODEL_NAMES[0], "tooltip_prompt": "Select the SAM2 Hiera model to download."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("sam_checkpoint_path", "sam_model_config_path",)
    FUNCTION = "download_model"
    CATEGORY = "SamMesh/Utils"

    def download_file(self, url, save_path, model_name):
        try:
            print(f"SamModelDownloader ({model_name}): Downloading {os.path.basename(save_path)} from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024

            from tqdm import tqdm
            with open(save_path, 'wb') as f, tqdm(
                desc=os.path.basename(save_path),
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    bar.update(size)
            print(f"SamModelDownloader ({model_name}): Download complete: {save_path}")
            return save_path
        except Exception as e:
            print(f"\033[31mError downloading {url} for {model_name}: {e}\033[0m")
            if os.path.exists(save_path):
                os.remove(save_path)
            raise

    def download_model(self, model_name: str):
        if model_name not in SAM_MODELS:
            raise ValueError(f"Selected model '{model_name}' is not defined.")

        model_info = SAM_MODELS[model_name]
        checkpoint_filename = model_info["checkpoint_filename"]
        config_filename = model_info["config_filename"]
        repo_id = model_info["repo_id"]
        config_url = model_info["config_url"]

        checkpoint_path = os.path.join(sam_model_dir, checkpoint_filename)
        config_path = os.path.join(sam_model_dir, config_filename)

        if not os.path.exists(checkpoint_path):
            print(f"SamModelDownloader ({model_name}): Checkpoint {checkpoint_filename} not found. Downloading...")
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=checkpoint_filename,
                    local_dir=sam_model_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True
                )
                print(f"SamModelDownloader ({model_name}): Checkpoint downloaded to {checkpoint_path}")
            except Exception as e:
                print(f"\033[31mError downloading checkpoint for {model_name} from Hugging Face: {e}\033[0m")
                raise
        else:
            print(f"SamModelDownloader ({model_name}): Checkpoint found: {checkpoint_path}")

        if not os.path.exists(config_path):
            print(f"SamModelDownloader ({model_name}): Config {config_filename} not found. Downloading...")
            self.download_file(config_url, config_path, model_name)
        else:
            print(f"SamModelDownloader ({model_name}): Config found: {config_path}")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Failed to download or locate checkpoint for {model_name}: {checkpoint_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Failed to download or locate config for {model_name}: {config_path}")

        return (checkpoint_path, config_path,)

MESH_EXTENSIONS = ['.glb', '.gltf', '.obj', '.ply', '.stl', '.3mf']

class SamMeshLoader:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = []
        try:
            for f in os.listdir(input_dir):
                if os.path.isfile(os.path.join(input_dir, f)):
                    _, ext = os.path.splitext(f)
                    if ext.lower() in MESH_EXTENSIONS:
                        files.append(f)
        except Exception as e:
            print(f"\033[93mWarning [SamMeshLoader]: Could not list input directory {input_dir}: {e}\033[0m")

        return {
            "required": {
                "mesh": (sorted(files) if files else ["No mesh files found"], {"tooltip_prompt": "Select a mesh file from the ComfyUI input directory."}),
            },
            "optional": {
                "normalize_mesh": ("BOOLEAN", {"default": False, "tooltip_prompt": "Scales and translates the mesh to fit within a unit cube centered at the origin."}),
                "process_mesh": ("BOOLEAN", {"default": True, "tooltip_prompt": "Enables Trimesh default processing. Disable if precise original topology is critical."}),
            }
        }

    RETURN_TYPES = ("SAM_MESH", "STRING",)
    RETURN_NAMES = ("sam_mesh", "loaded_mesh_path",)
    FUNCTION = "load_mesh"
    CATEGORY = "SamMesh"

    def load_mesh(self, mesh: str, normalize_mesh: bool = False, process_mesh: bool = True):
        if read_mesh is None:
            raise ImportError("samesh.data.loaders.read_mesh could not be imported.")

        if mesh == "No mesh files found":
            raise ValueError("No mesh files found in the ComfyUI input directory.")

        mesh_full_path = folder_paths.get_annotated_filepath(mesh)

        if not mesh_full_path or not os.path.exists(mesh_full_path):
            raise FileNotFoundError(f"Mesh file not found or path is invalid: {mesh_full_path} (from filename: {mesh})")

        print(f"SamMeshLoader: Loading mesh from: {mesh_full_path}")
        mesh_file_path = Path(mesh_full_path)

        try:
            loaded_mesh = read_mesh(mesh_file_path, norm=normalize_mesh, process=process_mesh)
            if loaded_mesh is None:
                raise ValueError(f"samesh.data.loaders.read_mesh returned None for {mesh_full_path}. Is the file valid?")

            print(f"SamMeshLoader Debug: Loaded mesh type: {type(loaded_mesh)}")
            if hasattr(loaded_mesh, 'visual'):
                print(f"SamMeshLoader Debug: Visual type: {type(loaded_mesh.visual)}")
                if hasattr(loaded_mesh.visual, 'material') and loaded_mesh.visual.material is not None:
                    print(f"SamMeshLoader Debug: Material type: {type(loaded_mesh.visual.material)}")
                if hasattr(loaded_mesh.visual, 'uv') and loaded_mesh.visual.uv is not None:
                    print(f"SamMeshLoader Debug: UVs are present. Count: {len(loaded_mesh.visual.uv)}")
            
            print(f"SamMeshLoader: Mesh loaded successfully. Vertices: {len(loaded_mesh.vertices)}, Faces: {len(loaded_mesh.faces)}")
            absolute_mesh_path = os.path.abspath(mesh_full_path)
            return (loaded_mesh, absolute_mesh_path,)
        except Exception as e:
            print(f"\033[31mError loading mesh {mesh_full_path}: {e}\033[0m")
            raise

class SamMeshSegmenter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sam_mesh": ("SAM_MESH",),
                "mesh_path": ("STRING", {"forceInput": True}),
                "sam_checkpoint_path": ("STRING", {"forceInput": True}),
                "sam_model_config_path": ("STRING", {"forceInput": True}),
                "output_directory": ("STRING", {"default": DEFAULT_OUTPUT_DIR, "tooltip_prompt": "Directory to save the face-to-label JSON file and intermediate files."}),
                "cache_directory": ("STRING", {"default": DEFAULT_CACHE_DIR, "tooltip_prompt": "Directory for samesh to cache rendering or computation results."}),
                "keep_texture": ("BOOLEAN", {"default": False, "tooltip_prompt": "If checked, attempts to preserve the original mesh texture."}),
            },
            "optional": {
                "target_labels": ("INT", {"default": -1, "min": -1, "max": 10000, "tooltip_prompt": "Desired number of distinct segments. -1 for automatic."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip_prompt": "Random seed for reproducible results."}),
            }
        }

    RETURN_TYPES = ("SAM_MESH", "STRING",)
    RETURN_NAMES = ("segmented_mesh", "face2label_path",)
    FUNCTION = "segment_mesh"
    CATEGORY = "SamMesh"

    def segment_mesh(self, sam_mesh: trimesh.Trimesh, mesh_path: str,
                     sam_checkpoint_path: str, sam_model_config_path: str,
                     output_directory: str, cache_directory: str,
                     keep_texture: bool, target_labels: int = -1,
                     seed: int = 0
                     ):

        if not isinstance(sam_mesh, trimesh.Trimesh):
            print(f"Warning: Input 'sam_mesh' is not a Trimesh object (got {type(sam_mesh)}).")
        if not mesh_path or not os.path.exists(mesh_path):
            mesh_path = os.path.abspath(mesh_path)
            if not os.path.exists(mesh_path):
                raise FileNotFoundError(f"Original mesh file path is invalid: {mesh_path}")
        else:
            mesh_path = os.path.abspath(mesh_path)

        if not sam_checkpoint_path or not os.path.exists(sam_checkpoint_path):
            raise FileNotFoundError(f"SAM checkpoint file not found: {sam_checkpoint_path}")
        if not sam_model_config_path or not os.path.exists(sam_model_config_path):
            raise FileNotFoundError(f"SAM model config file not found: {sam_model_config_path}")

        print(f"SamMeshSegmenter: Starting segmentation for: {mesh_path} via worker script.")
        os.makedirs(output_directory, exist_ok=True)
        os.makedirs(cache_directory, exist_ok=True)

        output_filename_prefix = "segmented"
        output_extension = "glb"
        capped_seed = seed % (2**32)

        cmd = [
            sys.executable,
            WORKER_SCRIPT_PATH,
            "--mesh_path", mesh_path,
            "--sam_checkpoint_path", sam_checkpoint_path,
            "--sam_model_config_path", sam_model_config_path,
            "--output_directory", output_directory,
            "--cache_directory", cache_directory,
            "--output_filename_prefix", output_filename_prefix,
            "--output_extension", output_extension,
            "--target_labels", str(target_labels),
        ]
        if keep_texture:
            cmd.append("--keep_texture")
        cmd.extend(["--seed", str(capped_seed)])

        env = os.environ.copy()
        if sys.platform == "win32" or os.name == "nt":
            env.pop("PYOPENGL_PLATFORM", None)
            env.pop("EGL_DEVICE_ID", None)

        try:
            comfy_base = folder_paths.base_path
            env['COMFYUI_BASE_PATH'] = comfy_base
        except AttributeError:
            env['COMFYUI_BASE_PATH'] = ''

        print(f"SamMeshSegmenter: Executing worker: {' '.join(cmd)}")
        try:
            samesh_worker_cwd = os.path.join(comfyui_samesh_node_dir, "samesh-main", "src")
            print(f"SamMeshSegmenter: Setting worker CWD to: {samesh_worker_cwd}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                cwd=samesh_worker_cwd
            )

            stdout_lines = []
            for line in process.stdout:
                print(line, end="", flush=True)
                stdout_lines.append(line)

            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd, output="".join(stdout_lines))

            last_line = "".join(stdout_lines).strip().splitlines()[-1]
            result_data = json.loads(last_line)
            final_output_mesh_path = result_data.get("output_mesh_path")
            final_output_json_path = result_data.get("face2label_path")

            if not final_output_mesh_path or "Error:" in final_output_mesh_path:
                raise RuntimeError(f"Worker script failed to produce mesh output: {final_output_mesh_path}")
            if not final_output_json_path or "Error:" in final_output_json_path:
                raise RuntimeError(f"Worker script failed to produce JSON output: {final_output_json_path}")

            print(f"SamMeshSegmenter: Worker finished successfully. Output JSON: {final_output_json_path}")
            if not os.path.exists(final_output_mesh_path):
                raise FileNotFoundError(f"Segmented mesh file not found: {final_output_mesh_path}")

            segmented_colored_mesh = trimesh.load(final_output_mesh_path, force='mesh')
            if not isinstance(segmented_colored_mesh, trimesh.Trimesh):
                raise TypeError(f"Loaded result is not a Trimesh object: {final_output_mesh_path}")

            return (segmented_colored_mesh, final_output_json_path,)
        except subprocess.CalledProcessError as e:
            print(f"\033[31mError: Worker script failed (return code {e.returncode}).\033[0m")
            raise RuntimeError("Segmentation worker script failed. Check logs for details.") from e
        except json.JSONDecodeError as e:
            print(f"\033[31mError: Could not parse JSON result from worker script.\033[0m")
            raise RuntimeError("Failed to parse result from segmentation worker.") from e
        except Exception as e:
            print(f"\033[31mError during segmentation execution: {e}\033[0m")
            import traceback
            traceback.print_exc()
            raise

class SamMeshExporter:
    RETURN_TYPES = ()
    FUNCTION = "export_parts"
    OUTPUT_NODE = True
    CATEGORY = "SamMesh/Export"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segmented_mesh": ("SAM_MESH",),
                "face2label_path": ("STRING", {"forceInput": True}),
                "output_filename": ("STRING", {"default": "holopart_input.glb", "tooltip_prompt": "Filename for the exported GLB scene."}),
            }
        }

    def export_parts(self, segmented_mesh: trimesh.Trimesh, face2label_path: str, output_filename: str):
        if not os.path.exists(face2label_path):
            raise FileNotFoundError(f"Face-to-label JSON file not found: {face2label_path}")

        output_base_dir = folder_paths.get_output_directory()
        if not output_filename.lower().endswith('.glb'):
            output_filename += '.glb'
            
        final_output_path = os.path.join(output_base_dir, output_filename)
        stem_name = os.path.splitext(output_filename)[0]
        
        # Dedicated folder to store each part as its own isolated .glb file
        parts_dir = os.path.join(output_base_dir, f"{stem_name}_parts")
        os.makedirs(parts_dir, exist_ok=True)

        print(f"SamMeshExporter: Preparing scene and parts folder at: {parts_dir}")

        try:
            with open(face2label_path, 'r') as f:
                face2label = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load face2label JSON: {e}")

        try:
            face_labels = {int(k): int(v) for k, v in face2label.items()}
            unique_labels = sorted(list(set(face_labels.values())))
            print(f"SamMeshExporter: Found {len(unique_labels)} unique segment labels.")
        except Exception as e:
            raise ValueError(f"Error processing face labels: {e}")

        base_mesh = segmented_mesh
        scene = trimesh.Scene()
        part_idx = 1

        has_texture_info = False
        if (hasattr(base_mesh, 'visual') and 
            isinstance(base_mesh.visual, trimesh.visual.TextureVisuals) and 
            hasattr(base_mesh.visual, 'uv') and 
            base_mesh.visual.uv is not None and 
            hasattr(base_mesh.visual, 'material') and 
            base_mesh.visual.material is not None and
            len(base_mesh.visual.uv) == len(base_mesh.vertices)):
            has_texture_info = True

        # Generate a distinct random color for each label so parts are visibly separated in 3D previewers
        rng = np.random.RandomState(42)
        label_colors = {
            lbl: rng.randint(40, 240, size=3).tolist() + [255]
            for lbl in unique_labels
        }

        for label in unique_labels:
            faces_for_label_mask = [idx for idx, lbl in face_labels.items() if lbl == label]

            if not faces_for_label_mask:
                continue

            # Skip tiny noise fragments (< 30 faces)
            if len(faces_for_label_mask) < 30:
                continue

            try:
                face_indices_for_label = np.array(faces_for_label_mask, dtype=np.int64)
                segment_faces = base_mesh.faces[face_indices_for_label]
                unique_vertex_indices = np.unique(segment_faces)
                segment_vertices = base_mesh.vertices[unique_vertex_indices]
                vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertex_indices)}
                new_segment_faces = np.array([
                    [vertex_map[v_idx] for v_idx in face]
                    for face in segment_faces
                ], dtype=np.int64)

                segment_mesh = trimesh.Trimesh(vertices=segment_vertices, faces=new_segment_faces, process=False)

                if has_texture_info:
                    try:
                        segment_uvs = base_mesh.visual.uv[unique_vertex_indices]
                        original_texture_image = getattr(base_mesh.visual.material, 'baseColorTexture', None) or getattr(base_mesh.visual.material, 'image_texture', None)
                        if original_texture_image is not None:
                            new_material = trimesh.visual.material.PBRMaterial(
                                baseColorTexture=original_texture_image,
                                metallicFactor=getattr(base_mesh.visual.material, 'metallicFactor', 0.0),
                                roughnessFactor=getattr(base_mesh.visual.material, 'roughnessFactor', 0.5)
                            )
                            segment_mesh.visual = trimesh.visual.TextureVisuals(
                                uv=segment_uvs,
                                material=new_material 
                            )
                        else:
                            color_rgba = np.array(label_colors[label], dtype=np.uint8)
                            segment_mesh.visual = trimesh.visual.ColorVisuals(
                                mesh=segment_mesh,
                                face_colors=np.tile(color_rgba, (len(new_segment_faces), 1))
                            )
                    except Exception:
                        color_rgba = np.array(label_colors[label], dtype=np.uint8)
                        segment_mesh.visual = trimesh.visual.ColorVisuals(
                            mesh=segment_mesh,
                            face_colors=np.tile(color_rgba, (len(new_segment_faces), 1))
                        )
                else:
                    # Color the segment with its unique label color
                    color_rgba = np.array(label_colors[label], dtype=np.uint8)
                    segment_mesh.visual = trimesh.visual.ColorVisuals(
                        mesh=segment_mesh,
                        face_colors=np.tile(color_rgba, (len(new_segment_faces), 1))
                    )

                node_name = f"part_{part_idx:03d}_label_{label}"
                
                # 1. Export as a standalone individual .glb file in the parts folder
                part_file_path = os.path.join(parts_dir, f"{node_name}.glb")
                segment_mesh.export(part_file_path, file_type='glb')

                # 2. Add as an independent node in the master Scene
                scene.add_geometry(segment_mesh, node_name=node_name, geom_name=node_name)
                part_idx += 1

            except Exception as e:
                print(f"\033[91mSamMeshExporter: Error processing label {label}: {e}\033[0m")

        if part_idx == 1:
            print("SamMeshExporter: No mesh parts were generated. Skipping scene export.")
            return {}

        try:
            print(f"SamMeshExporter: Exporting scene with {part_idx - 1} separate segments to {final_output_path}")
            scene.export(final_output_path, file_type='glb')
            print(f"SamMeshExporter: Standalone individual part files saved to: {parts_dir}")
        except Exception as e:
            print(f"\033[91mSamMeshExporter: Error exporting scene: {e}\033[0m")
            raise

        return {}

class SamMeshRenderer:
    @classmethod
    def INPUT_TYPES(s):
        if pyrender is None:
            return {"required": {"error": ("STRING", {"default": "pyrender not installed. Node disabled."})}}
        return {
            "required": {
                "mesh": ("SAM_MESH",),
                "render_resolution": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64, "tooltip_prompt": "Resolution for each rendered view."}),
                "background_color": ("STRING", {"default": "[0.1, 0.1, 0.1, 1.0]", "tooltip_prompt": "Background RGBA color as string list."}),
            },
            "optional": {
                "face2label_path": ("STRING", {"default": "", "forceInput": True, "tooltip_prompt": "Optional path to face2label.json."}),
                "force_segment_colors": ("BOOLEAN", {"default": True, "tooltip_prompt": "Render mesh with distinct segment colors."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rendered_views",)
    FUNCTION = "render_views"
    CATEGORY = "SamMesh/Utils"

    def _parse_color(self, color_str: str, default=(0.1, 0.1, 0.1, 1.0)):
        try:
            color = json.loads(color_str)
            if isinstance(color, list) and len(color) == 4 and all(isinstance(x, (int, float)) for x in color):
                if any(x > 1.0 for x in color):
                    return [max(0.0, min(255.0, x)) / 255.0 for x in color]
                else:
                    return [max(0.0, min(1.0, x)) for x in color]
            else:
                return list(default)
        except Exception:
            return list(default)

    def render_views(self, mesh: trimesh.Trimesh, render_resolution: int, background_color: str, face2label_path: str = "", force_segment_colors: bool = True):
        if pyrender is None:
            raise ImportError("pyrender is required for SamMeshRenderer but was not found.")

        bg_color = self._parse_color(background_color)
        working_mesh = mesh
        is_colored_copy = False

        if force_segment_colors and face2label_path and os.path.exists(face2label_path):
            try:
                mesh_copy = copy.deepcopy(mesh)
                with open(face2label_path, 'r') as f:
                    face2label_data = json.load(f)
                
                face_labels_map = {int(k): int(v) for k, v in face2label_data.items()}
                if not mesh_copy.faces.shape[0] == 0:
                    unique_labels = sorted(list(set(val for val in face_labels_map.values() if val is not None)))
                    if unique_labels:
                        label_to_color_idx_map = {label: i for i, label in enumerate(unique_labels)}
                        colors_for_labels = [trimesh.visual.random_color() for _ in range(len(unique_labels))]
                        default_face_color = np.array([128, 128, 128, 255], dtype=np.uint8)
                        new_face_colors = np.tile(default_face_color, (len(mesh_copy.faces), 1))

                        for face_idx, label_val in face_labels_map.items():
                            if label_val is not None and face_idx < len(mesh_copy.faces):
                                color_idx = label_to_color_idx_map.get(label_val)
                                if color_idx is not None:
                                    new_face_colors[face_idx] = colors_for_labels[color_idx]

                        if not isinstance(mesh_copy.visual, trimesh.visual.ColorVisuals):
                            mesh_copy.visual = trimesh.visual.ColorVisuals()
                        
                        mesh_copy.visual.face_colors = new_face_colors
                        working_mesh = mesh_copy
                        is_colored_copy = True
            except Exception as e:
                print(f"SamMeshRenderer: Warning - Failed to apply segment colors: {e}.")

        render_mesh = pyrender.Mesh.from_trimesh(working_mesh, smooth=not is_colored_copy)
        if render_mesh.primitives:
            primitive = render_mesh.primitives[0]
            if not (hasattr(primitive, 'color_0') and primitive.color_0 is not None) and not (hasattr(primitive, 'material') and primitive.material is not None):
                primitive.material = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=[0.8, 0.8, 0.8, 1.0],
                    metallicFactor=0.2,
                    roughnessFactor=0.6
                )

        scene = pyrender.Scene(bg_color=bg_color[:3], ambient_light=[0.2, 0.2, 0.2])
        mesh_node = scene.add(render_mesh, pose=np.eye(4), name="mesh")

        bounds = mesh.bounds
        if bounds is None:
            center = [0, 0, 0]
            distance = 2.0
        else:
            center = mesh.centroid
            scale = np.max(bounds[1] - bounds[0])
            distance = scale * 1.5

        aspect_ratio = 1.0
        yfov = np.pi / 4.0
        camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspect_ratio)
        camera_node = scene.add(camera, pose=np.eye(4))

        def look_at(eye, target, up):
            forward = np.subtract(target, eye)
            forward = forward / np.linalg.norm(forward)
            right = np.cross(forward, up)
            if np.linalg.norm(right) < 1e-6:
                right = np.cross(forward, [0, 1, 0]) if abs(forward[1]) < 0.99 else np.cross(forward, [1, 0, 0])
            right = right / np.linalg.norm(right)
            new_up = np.cross(right, forward)
            new_up = new_up / np.linalg.norm(new_up)

            cam_to_world = np.eye(4)
            cam_to_world[0, :3] = right
            cam_to_world[1, :3] = new_up
            cam_to_world[2, :3] = -forward
            cam_to_world[:3, 3] = eye
            return cam_to_world

        z_up = [0, 0, 1]
        poses = {
            "front": look_at(center + [0, -distance * 1.2, distance * 0.4], center, z_up),
            "right": look_at(center + [distance * 1.2, 0, distance * 0.4], center, z_up),
            "top":   look_at(center + [0, 0, distance * 1.5], center, [0, 1, 0]),
            "back":  look_at(center + [0, distance * 1.2, distance * 0.4], center, z_up),
        }

        renderer = pyrender.OffscreenRenderer(render_resolution, render_resolution)
        rendered_images = {}
        light_node = None

        view_keys = ["front", "right", "top", "back"]
        for key in view_keys:
            try:
                if light_node is not None and scene.has_node(light_node):
                    scene.remove_node(light_node)
                
                directional_light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
                light_node = scene.add(directional_light, pose=poses[key], name=f"light_for_{key}")
                
                scene.set_pose(camera_node, poses[key])
                color, depth = renderer.render(scene)
                rendered_images[key] = Image.fromarray(color, 'RGB')
            except Exception as e:
                rendered_images[key] = Image.new('RGB', (render_resolution, render_resolution), (50, 50, 50))

        grid_size = 1024
        img_per_view = grid_size // 2
        final_image = Image.new('RGB', (grid_size, grid_size))

        img_front = rendered_images["front"].resize((img_per_view, img_per_view), Image.LANCZOS)
        img_right = rendered_images["right"].resize((img_per_view, img_per_view), Image.LANCZOS)
        img_top = rendered_images["top"].resize((img_per_view, img_per_view), Image.LANCZOS)
        img_back = rendered_images["back"].resize((img_per_view, img_per_view), Image.LANCZOS)

        final_image.paste(img_front, (0, 0))
        final_image.paste(img_right, (img_per_view, 0))
        final_image.paste(img_top, (0, img_per_view))
        final_image.paste(img_back, (img_per_view, img_per_view))

        renderer.delete()
        image_np = np.array(final_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]
        return (image_tensor,)

NODE_CLASS_MAPPINGS = {
    "SamModelDownloader+SamMesh": SamModelDownloader,
    "SamMeshLoader+SamMesh": SamMeshLoader,
    "SamMeshSegmenter+SamMesh": SamMeshSegmenter,
    "SamMeshExporter+SamMesh": SamMeshExporter,
    "SamMeshRenderer+SamMesh": SamMeshRenderer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SamModelDownloader+SamMesh": "SAM Model Downloader (SamMesh)",
    "SamMeshLoader+SamMesh": "Load Mesh (SamMesh)",
    "SamMeshSegmenter+SamMesh": "Segment Mesh with SAM (SamMesh)",
    "SamMeshExporter+SamMesh": "Export Segmented Mesh for HoloPart (SamMesh)",
    "SamMeshRenderer+SamMesh": "Render Mesh Views (SamMesh)",
}