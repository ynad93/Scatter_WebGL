"""Lit-sphere-impostor point material for ScatterView particles.

pygfx ships only flat-shaded point materials (``PointsMaterial``,
``PointsGaussianBlobMaterial``, ``PointsMarkerMaterial``,
``PointsSpriteMaterial``) — none of them reproduce VisPy's
``Markers(spherical=True)`` look, which rasterizes each particle as a
lit sphere imposter (disc + per-fragment normal reconstruction + Phong
lighting).

This module adds that material.  The fragment shader converts the quad's
[-1, 1] local coordinate into a view-space sphere normal
``n = (u, v, sqrt(1 - u^2 - v^2))``, applies single-light Phong shading
(ambient + diffuse + optional specular), and writes the per-vertex
color modulated by that lighting.

Per-vertex position / color / size layout exactly matches pygfx's stock
points shader so the existing Geometry buffers can be reused.
"""

from __future__ import annotations

import pygfx as gfx
import wgpu
from pygfx.materials import PointsMaterial
from pygfx.objects import Points
from pygfx.renderers.wgpu import (
    Binding,
    BaseShader,
    register_wgpu_render_function,
)


class SphereImpostorMaterial(PointsMaterial):
    """PointsMaterial that renders each point as a Phong-lit sphere imposter.

    Layout-compatible with stock ``PointsMaterial``: reuses the same
    ``positions``, ``colors``, ``sizes`` geometry buffers.  Adds four
    lighting knobs on the material's uniform block.

    Parameters
    ----------
    light_dir : tuple[float, float, float]
        Light direction in view space (unit vector; looking toward the
        light).  Default ``(0.3, 0.3, 1.0)``.  Callers should normalize
        and push this each frame to keep the highlight stable while the
        camera orbits — see ``RenderEngine._update_light_direction``.
    ambient : float
        Ambient term in [0, 1].  Default 0.25.
    specular : float
        Specular term in [0, 1].  Default 0.3.
    shininess : float
        Phong specular exponent.  Default 32.0.
    """

    uniform_type = dict(
        PointsMaterial.uniform_type,
        light_dir="3xf4",
        ambient="f4",
        specular="f4",
        shininess="f4",
    )

    def __init__(
        self,
        *,
        light_dir=(0.3, 0.3, 1.0),
        ambient: float = 0.25,
        specular: float = 0.3,
        shininess: float = 32.0,
        size_mode: str = "vertex",
        size_space: str = "screen",
        color_mode: str = "vertex",
        **kwargs,
    ):
        super().__init__(
            size_mode=size_mode, size_space=size_space,
            color_mode=color_mode, **kwargs,
        )
        self.light_dir = light_dir
        self.ambient = ambient
        self.specular = specular
        self.shininess = shininess

    @property
    def light_dir(self):
        return tuple(self.uniform_buffer.data["light_dir"])

    @light_dir.setter
    def light_dir(self, value) -> None:
        self.uniform_buffer.data["light_dir"] = tuple(value)
        self.uniform_buffer.update_range(0, 1)

    @property
    def ambient(self) -> float:
        return float(self.uniform_buffer.data["ambient"])

    @ambient.setter
    def ambient(self, value: float) -> None:
        self.uniform_buffer.data["ambient"] = float(value)
        self.uniform_buffer.update_range(0, 1)

    @property
    def specular(self) -> float:
        return float(self.uniform_buffer.data["specular"])

    @specular.setter
    def specular(self, value: float) -> None:
        self.uniform_buffer.data["specular"] = float(value)
        self.uniform_buffer.update_range(0, 1)

    @property
    def shininess(self) -> float:
        return float(self.uniform_buffer.data["shininess"])

    @shininess.setter
    def shininess(self, value: float) -> None:
        self.uniform_buffer.data["shininess"] = float(value)
        self.uniform_buffer.update_range(0, 1)


# ---------------------------------------------------------------------------
# WGSL shader
# ---------------------------------------------------------------------------

_SHADER = r"""
{$ include 'pygfx.std.wgsl' $}


fn is_nan_f32(v: f32) -> bool {
    return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0;
}
fn is_finite_vec3(v: vec3<f32>) -> bool {
    return !is_nan_f32(v.x) && !is_nan_f32(v.y) && !is_nan_f32(v.z);
}


struct VertexInput {
    @builtin(vertex_index) index: u32,
};


@vertex
fn vs_main(in: VertexInput) -> Varyings {
    let screen_factor: vec2<f32> = u_stdinfo.logical_size.xy / 2.0;
    let l2p: f32 = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x;

    let index = i32(in.index);
    let node_index = index / 6;
    let vertex_index = index % 6;

    // Load position and project through the standard cam / proj stack.
    let pos_m = load_s_positions(node_index);
    let pos_w = u_wobject.world_transform * vec4<f32>(pos_m.xyz, 1.0);
    let pos_c = u_stdinfo.cam_transform * pos_w;
    let pos_n = u_stdinfo.projection_transform * pos_c;
    let pos_s = (pos_n.xy / pos_n.w + 1.0) * screen_factor;

    // Per-point size.
    $$ if size_mode == 'vertex'
        let size_ref = load_s_sizes(node_index);
    $$ else
        let size_ref = u_material.size;
    $$ endif

    // Convert size to logical pixels based on size_space.
    $$ if size_space == 'screen'
        let size_px = size_ref;
    $$ else
        // World units.  Project a 1000-px horizontal offset back to world
        // space to get the inverse world-per-pixel scale.
        let shift_factor = 1000.0;
        let pos_s_shifted = pos_s + vec2<f32>(shift_factor, 0.0);
        let pos_n_shifted = vec4<f32>(
            (pos_s_shifted / screen_factor - 1.0) * pos_n.w, pos_n.z, pos_n.w
        );
        let pos_w_shifted = (
            u_stdinfo.cam_transform_inv
            * u_stdinfo.projection_transform_inv
            * pos_n_shifted
        );
        let world_per_px = (1.0 / shift_factor) * distance(pos_w.xyz, pos_w_shifted.xyz);
        let size_px = size_ref / max(world_per_px, 1e-30);
    $$ endif

    let min_size_for_pixel = 1.415 / l2p;
    let half_size = 0.5 * max(size_px, min_size_for_pixel);

    // Billboard quad corners.
    var deltas = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );

    var delta = deltas[vertex_index];
    if (!is_finite_vec3(pos_m.xyz) || size_px <= 0.0) {
        delta = vec2<f32>(0.0, 0.0);
    }

    let delta_s = delta * half_size;
    let the_pos_s = pos_s + delta_s;
    let the_pos_n = vec4<f32>(
        (the_pos_s / screen_factor - 1.0) * pos_n.w, pos_n.z, pos_n.w
    );

    var varyings: Varyings;
    varyings.position = vec4<f32>(the_pos_n);
    varyings.world_pos = vec3<f32>(ndc_to_world_pos(the_pos_n));
    // Local disc coordinate in [-1, 1]; fragment reconstructs a sphere
    // normal from this.
    varyings.uv = vec2<f32>(delta);
    varyings.size_px = f32(size_px);

    $$ if color_mode == 'vertex'
        $$ if color_buffer_channels == 4
            varyings.color = vec4<f32>(load_s_colors(node_index));
        $$ elif color_buffer_channels == 3
            varyings.color = vec4<f32>(load_s_colors(node_index), 1.0);
        $$ else
            varyings.color = vec4<f32>(load_s_colors(node_index), 0.0, 0.0, 1.0);
        $$ endif
    $$ else
        varyings.color = vec4<f32>(u_material.color);
    $$ endif

    varyings.pick_idx = u32(node_index);
    varyings.elementIndex = u32(node_index);
    return varyings;
}


@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {
    {$ include 'pygfx.clipping_planes.wgsl' $}

    // Disc to sphere normal reconstruction in view space.  Camera
    // looks down -Z, so the visible hemisphere has z > 0.
    let uv = varyings.uv;
    let r2 = dot(uv, uv);
    if (r2 > 1.0) { discard; }

    let z = sqrt(max(1.0 - r2, 0.0));
    let normal_v = vec3<f32>(uv.x, uv.y, z);

    // Phong: ambient + diffuse + optional specular.
    let light_dir = normalize(u_material.light_dir);
    let n_dot_l = max(dot(normal_v, light_dir), 0.0);
    let ambient = u_material.ambient;
    var lit = varyings.color.rgb * (ambient + (1.0 - ambient) * n_dot_l);

    let view_dir = vec3<f32>(0.0, 0.0, 1.0);
    let halfv = normalize(light_dir + view_dir);
    let spec = pow(max(dot(normal_v, halfv), 0.0), u_material.shininess);
    lit = lit + vec3<f32>(u_material.specular * spec * n_dot_l);

    let alpha = clamp(varyings.color.a, 0.0, 1.0) * u_material.opacity;
    if (alpha <= 0.0) { discard; }
    do_alpha_test(alpha);

    // Antialias the silhouette by a one-pixel fade.
    let l2p: f32 = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x;
    let half_size_p = 0.5 * varyings.size_px * l2p;
    let r_p = sqrt(r2) * half_size_p;
    let edge_alpha = clamp(half_size_p - r_p + 0.5, 0.0, 1.0);

    let out_rgb = srgb2physical(lit);
    var out: FragmentOutput;
    out.color = vec4<f32>(out_rgb, alpha * edge_alpha);

    $$ if write_pick
    out.pick = (
        pick_pack(u32(u_wobject.global_id), 20) +
        pick_pack(varyings.pick_idx, 26)
    );
    $$ endif

    return out;
}
"""


@register_wgpu_render_function(Points, SphereImpostorMaterial)
class SphereImpostorShader(BaseShader):
    """Render function registration for ``SphereImpostorMaterial``.

    Only the combinations we actually use are implemented:
    ``size_mode="vertex"`` / ``"uniform"``, ``size_space="screen"`` /
    ``"world"``, and ``color_mode="vertex"`` / ``"uniform"``.
    """

    type = "render"

    def __init__(self, wobject):
        super().__init__(wobject)
        material: SphereImpostorMaterial = wobject.material
        geometry = wobject.geometry

        self["size_mode"] = str(material.size_mode).split(".")[-1]
        self["size_space"] = material.size_space

        color_mode = str(material.color_mode).split(".")[-1]
        if color_mode == "vertex":
            self["color_mode"] = "vertex"
            from pygfx.renderers.wgpu import nchannels_from_format
            self["color_buffer_channels"] = nchannels_from_format(geometry.colors.format)
        else:
            self["color_mode"] = "uniform"
            self["color_buffer_channels"] = 0

    def get_bindings(self, wobject, shared, scene):
        geometry = wobject.geometry
        material = wobject.material
        rbuffer = "buffer/read_only_storage"
        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
            Binding("s_positions", rbuffer, geometry.positions, "VERTEX"),
        ]
        if self["size_mode"] == "vertex":
            bindings.append(Binding("s_sizes", rbuffer, geometry.sizes, "VERTEX"))
        if self["color_mode"] == "vertex":
            bindings.append(Binding("s_colors", rbuffer, geometry.colors, "VERTEX"))

        bindings = {i: b for i, b in enumerate(bindings)}
        self.define_bindings(0, bindings)
        return {0: bindings}

    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        offset, size = wobject.geometry.positions.draw_range
        return {"indices": (size * 6, 1, offset * 6, 0)}

    def get_code(self):
        return _SHADER
