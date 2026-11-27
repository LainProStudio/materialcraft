use super::ATOMIC;
use crate::{CookContext, CookResult, NodeDefinition, node};
use craft_texture::{Image, Kind, Limits};
use rayon::prelude::*;

pub(super) fn definition() -> NodeDefinition {
    node("normal", "Normal", ATOMIC)
        .desc("Converts a height map to a tangent-space normal map.")
        .atomic()
        .gray_in("input", "Input")
        .float("intensity", "Intensity", 1.0, 0.0, 10.0)
        .boolean("opengl", "OpenGL normals", false)
        .boolean("height_alpha", "Height in alpha", true)
        .color_out()
        .cook(normal)
}

fn normal(context: &CookContext) -> CookResult {
    const BASE_RESOLUTION: f32 = 256.0;
    let height_map = context.gray_or(0, 0.0);
    let intensity = context.f("intensity") * context.w as f32 / BASE_RESOLUTION;

    Ok(vec![height_to_normal(&height_map, intensity, !context.b("opengl"), context.tile, context.b("height_alpha"))])
}

pub fn height_to_normal(height_map: &Image, intensity: f32, directx: bool, tile: bool, alpha_height: bool) -> Image {
    let (w, h) = (height_map.width(), height_map.height());
    let (wi, hi) = (w as usize, h as usize);
    let gray = height_map.to_gray();
    let src = gray.data();
    let y_sign = if directx { -intensity } else { intensity };
    let mut out = vec![0.0f32; wi * hi * 4];

    out.par_chunks_mut(wi * 4).enumerate().for_each(|(y, dst_row)| {
        let prev_y = if y > 0 {
            y - 1
        } else if tile {
            hi - 1
        } else {
            0
        };
        let curr = &src[y * wi..(y + 1) * wi];
        let prev = &src[prev_y * wi..(prev_y + 1) * wi];
        let mut left = if tile { curr[wi - 1] } else { curr[0] };
        for (x, dst) in dst_row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            // Backward differences place a wrapped discontinuity on its owning pixel
            let height = curr[x];
            let dx = height - left;
            let dy = height - prev[x];
            left = height;
            let nx = -dx * intensity;
            let ny = dy * y_sign;
            let inv_len = 0.5 / (nx * nx + ny * ny + 1.0).sqrt();
            let alpha = if alpha_height { height } else { 1.0 };
            dst[0] = nx * inv_len + 0.5;
            dst[1] = ny * inv_len + 0.5;
            dst[2] = inv_len + 0.5;
            dst[3] = alpha;
        }
    });

    let mut limits = Limits::default();
    limits.max_side = u32::MAX;
    limits.max_pixels = u64::MAX;
    limits.max_bytes = u64::MAX;
    Image::from_data(w, h, Kind::Color, out, &limits).unwrap_or_else(|_| Image::filled(w, h, Kind::Color, [0.5, 0.5, 1.0, 1.0]))
}
