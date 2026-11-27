use craft_texture::{Image, Kind, Limits};
use rayon::prelude::*;
use std::sync::Arc;

use super::{pattern_stamp::PatternStamp, region::Region, shape};
use crate::CookContext;

const MAX_SAMPLES: u64 = 256_000_000;
const MAX_STAMP_COPIES: u64 = 1_048_576;
const MIN_STAMP_SIZE: f32 = 1e-8;
// Parallelize horizontal row bands once pixel work justifies Rayon dispatch
const MIN_PARALLEL_SAMPLES: u64 = 32_768;

#[derive(Clone, Copy)]
struct PreparedStamp {
    center_x: f32,
    center_y: f32,
    cosine: f32,
    sine: f32,
    inv_sx: f32,
    inv_sy: f32,
    left: u32,
    right: u32,
    top: u32,
    bottom: u32,
    // Pre-multiplied per-channel tint * color_alpha * opacity (image_alpha applied separately when non-premultiplied)
    tint: [f32; 4],
    alpha_scale: f32,
    pattern: i32,
    variation: f32,
    blend: i32,
    filtering: i32,
    premultiplied: bool,
}

pub(super) fn render(context: &CookContext, stamps: &[PatternStamp], max_depth: u32) -> Result<Image, String> {
    let (width, height) = craft_texture::clamp_size(context.w, context.h);
    let kind = if context.b("color_mode") { Kind::Color } else { Kind::Gray };
    let mut background =
        context.input(0).map(|image| image.resize(width, height, context.tile)).unwrap_or_else(|| Image::filled(width, height, kind, context.v4("background")));
    background = if kind == Kind::Gray { background.to_gray() } else { background.to_color() };
    if stamps.is_empty() {
        return Ok(background);
    }

    let channels = kind.channels();
    let global_opacity = context.f("opacity").max(0.0);
    let weights = depth_weights(context.f("roughness"), max_depth)?;
    let integer_sampling = background.integer_sampling();
    let mut data = Arc::clone(background.shared_data());
    drop(background);

    let pixels = Arc::make_mut(&mut data);
    let mut budget = MAX_SAMPLES;
    let mut copies_left = MAX_STAMP_COPIES;
    let render_region = Region::new(context.v4("render_region"))?;
    let period = if context.tile { Region::new(context.v4("tiling_region"))?.period() } else { [1.0; 2] };

    // Pre-validate every stamp and expand its visible tiled copies in exact draw order
    let mut prepared = Vec::with_capacity(stamps.len());
    let mut total_samples: u64 = 0;

    for stamp in stamps {
        if !stamp.center.iter().chain(&stamp.size).chain(&stamp.color).chain([&stamp.rotation, &stamp.variation]).all(|value| value.is_finite()) {
            return Err("pattern parameters must be finite".into());
        }

        if stamp.size.iter().any(|size| size.abs() < MIN_STAMP_SIZE) {
            continue;
        }

        if !matches!(stamp.blend, 0..=2) {
            return Err(format!("unsupported pattern blending mode {}", stamp.blend));
        }

        if !(2..=18).contains(&stamp.pattern) && !(256..256 + crate::MAX_IMAGE_INPUTS as i32).contains(&stamp.pattern) {
            return Err(format!("unsupported pattern {}", stamp.pattern));
        }

        let opacity = global_opacity * weights.get(stamp.depth as usize).copied().ok_or("invalid pattern depth")?;
        if !opacity.is_finite() {
            return Err("pattern roughness overflowed its depth opacity".into());
        }

        // Scalar luminosity weights both RGB and alpha, unlike straight RGBA tint
        let source_alpha = if stamp.scalar_color { stamp.color[0] } else { stamp.color[3] };
        let color_rgb_alpha = if kind == Kind::Color { stamp.color[3] } else { 1.0 };
        let tint = [
            stamp.color[0] * color_rgb_alpha * opacity,
            stamp.color[1] * color_rgb_alpha * opacity,
            stamp.color[2] * color_rgb_alpha * opacity,
            source_alpha * opacity,
        ];
        let alpha_scale = source_alpha * opacity;

        let (sine, cosine) = (stamp.rotation * std::f32::consts::TAU).sin_cos();
        let radius = [
            (stamp.size[0] * cosine).abs() * 0.5 + (stamp.size[1] * sine).abs() * 0.5,
            (stamp.size[0] * sine).abs() * 0.5 + (stamp.size[1] * cosine).abs() * 0.5,
        ];
        let base_center = if context.tile { std::array::from_fn(|axis| stamp.center[axis].rem_euclid(period[axis])) } else { stamp.center };
        let repeats = if context.tile { render_region.repeats(base_center, radius, period)? } else { [[0, 0]; 2] };
        let count = |axis: usize| (i64::from(repeats[axis][1]) - i64::from(repeats[axis][0]) + 1).max(0) as u64;
        let copies = count(0).checked_mul(count(1)).ok_or("pattern repetition exceeds its budget")?;
        copies_left = copies_left.checked_sub(copies).ok_or("pattern repetition exceeds its budget")?;

        let inv_sx = 2.0 / stamp.size[0];
        let inv_sy = 2.0 / stamp.size[1];

        for repeat_y in repeats[1][0]..=repeats[1][1] {
            for repeat_x in repeats[0][0]..=repeats[0][1] {
                let center = [base_center[0] + repeat_x as f32 * period[0], base_center[1] + repeat_y as f32 * period[1]];
                let [[left, right], [top, bottom]] = render_region.bounds(center, radius, [width, height]);
                let samples = u64::from(right.saturating_sub(left)) * u64::from(bottom.saturating_sub(top));
                budget = budget.checked_sub(samples).ok_or("pattern rendering exceeded its pixel budget")?;
                if samples == 0 {
                    continue;
                }
                total_samples = total_samples.saturating_add(samples);
                prepared.push(PreparedStamp {
                    center_x: center[0],
                    center_y: center[1],
                    cosine,
                    sine,
                    inv_sx,
                    inv_sy,
                    left,
                    right,
                    top,
                    bottom,
                    tint,
                    alpha_scale,
                    pattern: stamp.pattern,
                    variation: stamp.variation,
                    blend: stamp.blend,
                    filtering: stamp.filtering,
                    premultiplied: stamp.premultiplied,
                });
            }
        }
    }

    if prepared.is_empty() {
        return Image::from_data(width, height, kind, data, &Limits::default())
            .map(|image| image.with_integer_sampling(integer_sampling))
            .map_err(|e| e.to_string());
    }

    // Precompute exact pixel-center coordinates once (bit-identical to Region::point)
    let xs: Vec<f32> = (0..width).map(|x| render_region.point([x, 0], [width, height])[0]).collect();
    let ys: Vec<f32> = (0..height).map(|y| render_region.point([0, y], [width, height])[1]).collect();

    if total_samples < MIN_PARALLEL_SAMPLES || height < 32 {
        for stamp in &prepared {
            rasterize_stamp(stamp, stamp.top, stamp.bottom, 0, width as usize, kind, &xs, &ys, context.inputs, pixels)?;
        }
    } else {
        let band_rows: u32 = if height >= 512 { 32 } else { 16 };
        let num_bands = height.div_ceil(band_rows) as usize;
        let mut band_bins: Vec<Vec<u32>> = vec![Vec::new(); num_bands];
        for (idx, stamp) in prepared.iter().enumerate() {
            let b_start = (stamp.top / band_rows) as usize;
            let b_end = ((stamp.bottom - 1) / band_rows) as usize;
            for bin in band_bins.iter_mut().take(b_end + 1).skip(b_start) {
                bin.push(idx as u32);
            }
        }

        let row_stride = width as usize * channels;
        let band_floats = band_rows as usize * row_stride;
        pixels.par_chunks_mut(band_floats).enumerate().try_for_each(|(band_idx, band_pixels)| -> Result<(), String> {
            let band_top = band_idx as u32 * band_rows;
            let band_bottom = (band_top + band_rows).min(height);
            let Some(bin) = band_bins.get(band_idx) else { return Ok(()) };
            for &stamp_idx in bin {
                if let Some(stamp) = prepared.get(stamp_idx as usize) {
                    let y0 = stamp.top.max(band_top);
                    let y1 = stamp.bottom.min(band_bottom);
                    if y0 < y1 {
                        rasterize_stamp(stamp, y0, y1, band_top, width as usize, kind, &xs, &ys, context.inputs, band_pixels)?;
                    }
                }
            }
            Ok(())
        })?;
    }

    Image::from_data(width, height, kind, data, &Limits::default()).map(|image| image.with_integer_sampling(integer_sampling)).map_err(|e| e.to_string())
}

#[allow(clippy::too_many_arguments)]
fn rasterize_stamp(
    s: &PreparedStamp,
    y_start: u32,
    y_end: u32,
    band_top: u32,
    width: usize,
    kind: Kind,
    xs: &[f32],
    ys: &[f32],
    inputs: &[Option<Image>],
    band_pixels: &mut [f32],
) -> Result<(), String> {
    let x_start = s.left as usize;
    let x_end = s.right as usize;
    let axis_aligned = s.sine == 0.0;

    if s.pattern >= 256 {
        let index = (s.pattern - 256) as usize + 1;
        let input = inputs.get(index).and_then(Option::as_ref);
        return rasterize_image_stamp(s, y_start, y_end, band_top, width, kind, xs, ys, x_start, x_end, input, band_pixels);
    }

    if kind == Kind::Gray {
        let tint = s.tint[0];
        let blend_max = s.blend == 1 || s.blend == 2;
        if s.pattern == 2 && axis_aligned {
            // Fast path for unrotated solid square stamps (white_noise_fast, quadrant fills)
            let c_inv_sx = s.cosine * s.inv_sx;
            let c_inv_sy = s.cosine * s.inv_sy;
            for y in y_start..y_end {
                let local_y = (ys[y as usize] - s.center_y) * c_inv_sy;
                if local_y <= -1.0 || local_y > 1.0 {
                    continue;
                }
                let row_offset = (y - band_top) as usize * width;
                let row = &mut band_pixels[row_offset + x_start..row_offset + x_end];
                let x_slice = &xs[x_start..x_end];
                if blend_max {
                    for (dst, &px) in row.iter_mut().zip(x_slice) {
                        let local_x = (px - s.center_x) * c_inv_sx;
                        if local_x > -1.0 && local_x <= 1.0 {
                            let next = dst.max(tint);
                            if !next.is_finite() {
                                return Err("pattern accumulation overflowed".into());
                            }
                            *dst = next;
                        }
                    }
                } else {
                    for (dst, &px) in row.iter_mut().zip(x_slice) {
                        let local_x = (px - s.center_x) * c_inv_sx;
                        if local_x > -1.0 && local_x <= 1.0 {
                            let next = *dst + tint;
                            if !next.is_finite() {
                                return Err("pattern accumulation overflowed".into());
                            }
                            *dst = next;
                        }
                    }
                }
            }
            return Ok(());
        }

        for y in y_start..y_end {
            let dy = ys[y as usize] - s.center_y;
            let dy_sin = dy * s.sine;
            let dy_cos = dy * s.cosine;
            if axis_aligned {
                let local_y = dy_cos * s.inv_sy;
                if local_y <= -1.0 || local_y > 1.0 {
                    continue;
                }
            }
            let row_offset = (y - band_top) as usize * width;
            let row = &mut band_pixels[row_offset + x_start..row_offset + x_end];
            let x_slice = &xs[x_start..x_end];
            if blend_max {
                for (dst, &px) in row.iter_mut().zip(x_slice) {
                    let dx = px - s.center_x;
                    let lx = (dx * s.cosine + dy_sin) * s.inv_sx;
                    let ly = (-dx * s.sine + dy_cos) * s.inv_sy;
                    if lx <= -1.0 || lx > 1.0 || ly <= -1.0 || ly > 1.0 {
                        continue;
                    }
                    let cov = shape::coverage_validated(s.pattern, lx, ly, s.variation);
                    let next = dst.max(cov * tint);
                    if !next.is_finite() {
                        return Err("pattern accumulation overflowed".into());
                    }
                    *dst = next;
                }
            } else {
                for (dst, &px) in row.iter_mut().zip(x_slice) {
                    let dx = px - s.center_x;
                    let lx = (dx * s.cosine + dy_sin) * s.inv_sx;
                    let ly = (-dx * s.sine + dy_cos) * s.inv_sy;
                    if lx <= -1.0 || lx > 1.0 || ly <= -1.0 || ly > 1.0 {
                        continue;
                    }
                    let cov = shape::coverage_validated(s.pattern, lx, ly, s.variation);
                    let next = *dst + cov * tint;
                    if !next.is_finite() {
                        return Err("pattern accumulation overflowed".into());
                    }
                    *dst = next;
                }
            }
        }
        return Ok(());
    }

    // Color built-in shape
    let t = s.tint;
    let alpha_scale = s.alpha_scale;
    for y in y_start..y_end {
        let dy = ys[y as usize] - s.center_y;
        let dy_sin = dy * s.sine;
        let dy_cos = dy * s.cosine;
        if axis_aligned {
            let local_y = dy_cos * s.inv_sy;
            if local_y <= -1.0 || local_y > 1.0 {
                continue;
            }
        }
        let row_offset = (y - band_top) as usize * width * 4;
        let row = &mut band_pixels[row_offset + x_start * 4..row_offset + x_end * 4];
        let x_slice = &xs[x_start..x_end];
        match s.blend {
            1 => {
                for (dst, &px) in row.as_chunks_mut::<4>().0.iter_mut().zip(x_slice) {
                    let dx = px - s.center_x;
                    let lx = (dx * s.cosine + dy_sin) * s.inv_sx;
                    let ly = (-dx * s.sine + dy_cos) * s.inv_sy;
                    if lx <= -1.0 || lx > 1.0 || ly <= -1.0 || ly > 1.0 {
                        continue;
                    }
                    let cov = shape::coverage_validated(s.pattern, lx, ly, s.variation);
                    let keep = 1.0 - (cov * alpha_scale).clamp(0.0, 1.0);
                    for c in 0..4 {
                        let next = cov * t[c] + dst[c] * keep;
                        if !next.is_finite() {
                            return Err("pattern accumulation overflowed".into());
                        }
                        dst[c] = next;
                    }
                }
            }
            2 => {
                for (dst, &px) in row.as_chunks_mut::<4>().0.iter_mut().zip(x_slice) {
                    let dx = px - s.center_x;
                    let lx = (dx * s.cosine + dy_sin) * s.inv_sx;
                    let ly = (-dx * s.sine + dy_cos) * s.inv_sy;
                    if lx <= -1.0 || lx > 1.0 || ly <= -1.0 || ly > 1.0 {
                        continue;
                    }
                    let cov = shape::coverage_validated(s.pattern, lx, ly, s.variation);
                    for c in 0..4 {
                        let next = dst[c].max(cov * t[c]);
                        if !next.is_finite() {
                            return Err("pattern accumulation overflowed".into());
                        }
                        dst[c] = next;
                    }
                }
            }
            _ => {
                for (dst, &px) in row.as_chunks_mut::<4>().0.iter_mut().zip(x_slice) {
                    let dx = px - s.center_x;
                    let lx = (dx * s.cosine + dy_sin) * s.inv_sx;
                    let ly = (-dx * s.sine + dy_cos) * s.inv_sy;
                    if lx <= -1.0 || lx > 1.0 || ly <= -1.0 || ly > 1.0 {
                        continue;
                    }
                    let cov = shape::coverage_validated(s.pattern, lx, ly, s.variation);
                    for c in 0..4 {
                        let next = dst[c] + cov * t[c];
                        if !next.is_finite() {
                            return Err("pattern accumulation overflowed".into());
                        }
                        dst[c] = next;
                    }
                }
            }
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn rasterize_image_stamp(
    s: &PreparedStamp,
    y_start: u32,
    y_end: u32,
    band_top: u32,
    width: usize,
    kind: Kind,
    xs: &[f32],
    ys: &[f32],
    x_start: usize,
    x_end: usize,
    input: Option<&Image>,
    band_pixels: &mut [f32],
) -> Result<(), String> {
    let channels = kind.channels();
    let t = s.tint;
    let alpha_scale = s.alpha_scale;
    let mul_img_alpha = !s.premultiplied;
    let nearest = s.filtering == 0;
    let in_w = input.map_or(0.0, |img| img.width() as f32);
    let in_h = input.map_or(0.0, |img| img.height() as f32);

    for y in y_start..y_end {
        let dy = ys[y as usize] - s.center_y;
        let dy_sin = dy * s.sine;
        let dy_cos = dy * s.cosine;
        if s.sine == 0.0 {
            let local_y = dy_cos * s.inv_sy;
            if local_y <= -1.0 || local_y > 1.0 {
                continue;
            }
        }
        let row_offset = (y - band_top) as usize * width * channels;
        let row = &mut band_pixels[row_offset + x_start * channels..row_offset + x_end * channels];
        let x_slice = &xs[x_start..x_end];

        for (dst, &px) in row.chunks_exact_mut(channels).zip(x_slice) {
            let dx = px - s.center_x;
            let lx = (dx * s.cosine + dy_sin) * s.inv_sx;
            let ly = (-dx * s.sine + dy_cos) * s.inv_sy;
            if lx <= -1.0 || lx > 1.0 || ly <= -1.0 || ly > 1.0 {
                continue;
            }

            let pattern = match input {
                Some(img) => {
                    let u = lx * 0.5 + 0.5;
                    let v = ly * 0.5 + 0.5;
                    if nearest { img.get((u * in_w) as i64, (v * in_h) as i64, false) } else { img.sample(u, v, false) }
                }
                None => [0.0; 4],
            };

            let img_a = if mul_img_alpha { pattern[3] } else { 1.0 };
            if kind == Kind::Gray {
                let value = pattern[0] * t[0] * img_a;
                let next = if s.blend == 1 || s.blend == 2 { dst[0].max(value) } else { dst[0] + value };
                if !next.is_finite() {
                    return Err("pattern accumulation overflowed".into());
                }
                dst[0] = next;
            } else {
                let keep = if s.blend == 1 { 1.0 - (pattern[3] * alpha_scale).clamp(0.0, 1.0) } else { 1.0 };
                for c in 0..4 {
                    let ia = if c < 3 { img_a } else { 1.0 };
                    let value = pattern[c] * t[c] * ia;
                    let next = match s.blend {
                        1 => value + dst[c] * keep,
                        2 => dst[c].max(value),
                        _ => dst[c] + value,
                    };
                    if !next.is_finite() {
                        return Err("pattern accumulation overflowed".into());
                    }
                    dst[c] = next;
                }
            }
        }
    }

    Ok(())
}

fn depth_weights(roughness: f32, max_depth: u32) -> Result<Vec<f32>, String> {
    // Match the depth allocation cap to the 4096-node acyclic traversal limit
    if max_depth >= 4096 || !roughness.is_finite() || roughness < 0.0 {
        return Err("pattern roughness must be finite and nonnegative, with depth below 4096".into());
    }

    let mut weights = vec![1.0_f64; max_depth as usize + 1];

    // Zero is the native no-attenuation mode, not the limit of the normalized geometric weights
    if roughness == 0.0 {
        return Ok(weights.into_iter().map(|value| value as f32).collect());
    }

    // Normalize depth weights across drawing and non-drawing ancestors
    // Start at the largest weight to avoid overflow
    let mut weight = 1.0;
    let ratio = f64::from(roughness);
    if roughness <= 1.0 {
        for value in &mut weights {
            *value = weight;
            weight *= ratio;
        }
    } else {
        for value in weights.iter_mut().rev() {
            *value = weight;
            weight /= ratio;
        }
    }

    let sum: f64 = weights.iter().sum();
    Ok(weights.into_iter().map(|value| (value / sum) as f32).collect())
}
