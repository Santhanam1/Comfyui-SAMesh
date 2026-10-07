import os
import sys

# On Windows, purge EGL settings from os.environ so PyOpenGL falls back to native WGL
if sys.platform == "win32" or os.name == "nt":
    os.environ.pop("PYOPENGL_PLATFORM", None)
    os.environ.pop("EGL_DEVICE_ID", None)
else:
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["EGL_DEVICE_ID"] = "-1"

### START VOODOO ###
# Dark incantation for disabling anti-aliasing in pyrender (if needed)
import OpenGL.GL
antialias_active = False
old_gl_enable = OpenGL.GL.glEnable
def new_gl_enable(value):
    if not antialias_active and value == OpenGL.GL.GL_MULTISAMPLE:
        OpenGL.GL.glDisable(value)
    else:
        old_gl_enable(value)
OpenGL.GL.glEnable = new_gl_enable
import pyrender
### END VOODOO ###

import cv2
import numpy as np
import torch
from numpy.random import RandomState
from PIL import Image
from pyrender.shader_program import ShaderProgramCache as DefaultShaderCache
from trimesh import Trimesh, Scene
from omegaconf import OmegaConf
from tqdm import tqdm

from samesh.data.common import NumpyTensor
from samesh.data.loaders import scene2mesh
from samesh.utils.cameras import HomogeneousTransform, sample_view_matrices, sample_view_matrices_polyhedra
from samesh.utils.math import range_norm
from samesh.utils.mesh import duplicate_verts
from samesh.renderer.shader_programs import *


def colormap_faces(faces, background=np.array([255, 255, 255])) -> Image.Image:
    """
    Given a face id map, color each face with a random color.
    """
    palette = RandomState(0).randint(0, 255, (np.max(faces + 2), 3))
    palette[0] = background
    image = palette[faces + 1, :].astype(np.uint8)
    return Image.fromarray(image)


def colormap_norms(norms, background=np.array([255, 255, 255])) -> Image.Image:
    """
    Given a normal map, color each normal with a color.
    """
    norms = (norms + 1) / 2
    norms = (norms * 255).astype(np.uint8)
    return Image.fromarray(norms)


DEFAULT_CAMERA_PARAMS = {'fov': 60, 'znear': 0.01, 'zfar': 16}


class Renderer:
    def __init__(self, config: OmegaConf):
        self.config = config
        self.renderer = pyrender.OffscreenRenderer(*config.target_dim)
        self.shaders = {
            'default': DefaultShaderCache(),
            'normals': NormalShaderCache(),
            'faceids': FaceidShaderCache(),
            'barycnt': BarycentricShaderCache(),
        }

    def set_object(self, source: Trimesh | Scene, smooth=False):
        if isinstance(source, Scene):
            self.tmesh = scene2mesh(source)
            self.scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0])
            for name, geom in source.geometry.items():
                if name in source.graph:
                    pose, _ = source.graph[name]
                else:
                    pose = None
                self.scene.add(pyrender.Mesh.from_trimesh(geom, smooth=smooth), pose=pose)
        
        elif isinstance(source, Trimesh):
            self.tmesh = source
            self.scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0])
            self.scene.add(pyrender.Mesh.from_trimesh(source, smooth=smooth))

        else:
            raise ValueError(f'Invalid source type {type(source)}')
        
        # Rearrange mesh for faceid rendering
        self.tmesh_faceid = duplicate_verts(self.tmesh)
        self.scene_faceid = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0])
        self.scene_faceid.add(
            pyrender.Mesh.from_trimesh(self.tmesh_faceid, smooth=smooth)
        )

    def set_camera(self, camera_params: dict = None):
        self.camera_params = camera_params or dict(DEFAULT_CAMERA_PARAMS)
        self.camera_params['yfov'] = self.camera_params.get('yfov', self.camera_params.pop('fov'))
        self.camera_params['yfov'] = self.camera_params['yfov'] * np.pi / 180.0
        self.camera = pyrender.PerspectiveCamera(**self.camera_params)
        
        self.camera_node        = self.scene.add(self.camera)
        self.camera_node_faceid = self.scene_faceid.add(self.camera)
        
    def render(
        self, 
        pose: np.ndarray, 
        lightdir=np.array([0.0, 0.0, 1.0]), uv_map=False, interpolate_norms=True, blur_matte=False
    ) -> dict:
        self.scene.set_pose(self.camera_node, pose)
        self.scene_faceid.set_pose(self.camera_node_faceid, pose)

        def render(shader: str, scene):
            self.renderer._renderer._program_cache = self.shaders[shader]
            return self.renderer.render(scene)
        
        if uv_map:
            raw_color, raw_depth = render('default', self.scene)
        raw_norms, raw_depth = render('normals', self.scene)
        raw_faces, raw_depth = render('faceids', self.scene_faceid)
        raw_bcent, raw_depth = render('barycnt', self.scene_faceid)

        def render_norms(norms) -> np.ndarray:
            return np.clip((norms / 255.0 - 0.5) * 2, -1, 1)

        def render_depth(depth, offset=2.8, alpha=0.8) -> np.ndarray:
            return np.where(depth > 0, alpha * (1.0 - range_norm(depth, offset=offset)), 1)

        def render_faces(faces) -> np.ndarray:
            faces = faces.astype(np.int32)
            faces = faces[:, :, 0] * 65536 + faces[:, :, 1] * 256 + faces[:, :, 2]
            faces[faces == (256 ** 3 - 1)] = -1
            return faces

        def render_bcent(bcent) -> np.ndarray:
            return np.clip(bcent / 255.0, 0, 1)

        def render_matte(
            norms,
            depth,
            faces,
            bcent,
            alpha=0.5, beta=0.25, gaussian_kernel_width=5, gaussian_sigma=1,
        ) -> np.ndarray:
            interpolated_norms = norms.copy()
            if interpolate_norms:
                num_faces = self.tmesh.faces.shape[0]
                valid_faces_mask = (faces >= 0) & (faces < num_faces)
                
                if np.any(valid_faces_mask):
                    valid_pixel_indices = np.where(valid_faces_mask.reshape(-1))[0]
                    valid_face_values = faces.reshape(-1)[valid_pixel_indices]
                    
                    verts_index = self.tmesh.faces[valid_face_values]
                    verts_norms = self.tmesh.vertex_normals[verts_index]
                    
                    valid_bcent = bcent.reshape(-1, 3)[valid_pixel_indices]
                    calculated_norms_flat = np.sum(verts_norms * valid_bcent[:, :, None], axis=1)
                    interpolated_norms.reshape(-1, 3)[valid_pixel_indices] = calculated_norms_flat

            diffuse = np.sum(interpolated_norms * lightdir, axis=2)
            diffuse = np.clip(diffuse, -1, 1)
            matte = 255 * (diffuse[:, :, None] * alpha + beta)
            matte = np.where(depth[:, :, None] > 0, matte, 255)
            matte = np.clip(matte, 0, 255).astype(np.uint8)
            matte = np.repeat(matte, 3, axis=2)
            
            if blur_matte:
                matte = (faces == -1)[:, :, None] * matte + \
                        (faces != -1)[:, :, None] * cv2.GaussianBlur(matte, (gaussian_kernel_width, gaussian_kernel_width), gaussian_sigma)
            return matte 

        norms = render_norms(raw_norms)
        depth = render_depth(raw_depth)
        faces = render_faces(raw_faces)
        bcent = render_bcent(raw_bcent)
        matte = raw_color if uv_map else render_matte(norms, raw_depth, faces, bcent)

        return {'norms': norms, 'depth': depth, 'matte': matte, 'faces': faces}


def render_multiview(
    renderer: Renderer,
    camera_generation_method='sphere',
    renderer_args: dict=None,
    sampling_args: dict=None,
    lighting_args: dict=None, 
    lookat_position=np.array([0, 0, 0]),
    verbose=True,
) -> list[Image.Image]:
    lookat_position_torch = torch.from_numpy(lookat_position)
    if camera_generation_method == 'sphere':
        views = sample_view_matrices(lookat_position=lookat_position_torch, **sampling_args).numpy()
    else:
        views = sample_view_matrices_polyhedra(camera_generation_method, lookat_position=lookat_position_torch, **sampling_args).numpy()
    
    def compute_lightdir(pose: np.ndarray) -> np.ndarray:
        lightdir = pose[:3, 3] - lookat_position
        return lightdir / np.linalg.norm(lightdir)

    renders = []
    if verbose:
        views = tqdm(views, 'Rendering Multiviews...')
    for pose in views:
        outputs = renderer.render(pose, lightdir=compute_lightdir(pose), **renderer_args)
        outputs['matte'] = Image.fromarray(outputs['matte'])
        outputs['poses'] = pose
        renders.append(outputs)
    return {
        name: [render[name] for render in renders] for name in renders[0].keys()
    }