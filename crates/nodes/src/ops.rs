//! Image operations shared by many nodes: Gaussian blur, Euclidean distance transform, slopes,
//! color space helpers and blend modes.

use craft_texture::{Image, Kind, Limits};
use rayon::prelude::*;

/// Gaussian blur with standard deviation `sigma` (pixels). Small sigmas use a sampled kernel;
/// larger ones three box passes (a close Gaussian approximation in O(1) per pixel).
pub fn gaussian(image: &Image, sigma: f32, tile: bool) -> Image {
    const MIN_SIGMA_PIXELS: f32 = 0.05;
    const BOX_BLUR_MIN_SIGMA_PIXELS: f32 = 3.0;
    const KERNEL_RADIUS_SIGMAS: f32 = 3.0;

    if sigma.is_nan() || sigma <= MIN_SIGMA_PIXELS || !sigma.is_finite() {
        return image.clone();
    }

    let sigma = sigma.min(image.width().max(image.height()) as f32);
    if sigma < BOX_BLUR_MIN_SIGMA_PIXELS {
        let radius = (sigma * KERNEL_RADIUS_SIGMAS).ceil() as i64;
        let kernel: Vec<f32> = (-radius..=radius).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
        let kernel_sum: f32 = kernel.iter().sum();
        let kernel: Vec<f32> = kernel.iter().map(|v| v / kernel_sum).collect();
        let pass = |src: &Image, horizontal: bool| convolve_1d(src, &kernel, radius, horizontal, tile);
        return pass(&pass(image, true), false);
    }

    // Box widths for three passes (Kovesi, "Fast almost-Gaussian filtering")
    let pass_count = 3.0f32;
    let lower_width = ((12.0 * sigma * sigma / pass_count + 1.0).sqrt()).floor() as i64;
    let lower_width = if lower_width % 2 == 0 { lower_width - 1 } else { lower_width };
    let upper_width = lower_width + 2;
    let lower_passes = ((12.0 * sigma * sigma - pass_count * (lower_width * lower_width) as f32 - 4.0 * pass_count * lower_width as f32 - 3.0 * pass_count)
        / (-4.0 * lower_width as f32 - 4.0))
        .round() as i64;
    let mut output = image.clone();
    for i in 0..3 {
        let radius = (if i < lower_passes { lower_width } else { upper_width } - 1) / 2;
        output = box_1d(&box_1d(&output, radius.max(0), true, tile), radius.max(0), false, tile);
    }

    output
}

/// The default limits, raised to admit `src`, so a pass over an image made with raised limits
/// returns an image of the same size.
fn limits_for(src: &Image) -> Limits {
    let mut limits = Limits::default();
    limits.max_side = limits.max_side.max(src.width()).max(src.height());
    limits.max_pixels = limits.max_pixels.max(u64::from(src.width()) * u64::from(src.height()));
    limits.max_bytes = limits.max_bytes.max(size_of_val(src.data()) as u64);
    limits
}

fn convolve_1d(src: &Image, kernel: &[f32], radius: i64, horizontal: bool, tile: bool) -> Image {
    let channels = src.channels();
    Image::try_generate(src.width(), src.height(), src.kind(), &limits_for(src), |x, y| {
        let mut accumulator = [0.0f32; 4];
        for (i, weight) in kernel.iter().enumerate() {
            let offset = i as i64 - radius;
            let pixel = if horizontal { src.get(x as i64 + offset, y as i64, tile) } else { src.get(x as i64, y as i64 + offset, tile) };
            for channel in 0..channels.clamp(1, 4) {
                accumulator[channel] += pixel[channel] * weight;
            }
        }
        if channels == 1 { [accumulator[0], 0.0, 0.0, 0.0] } else { accumulator }
    })
    .unwrap_or_else(|_| src.clone())
}

/// Cache-blocked parallel transpose of an interleaved `w × h × channels` float buffer to `h × w × channels`.
pub(crate) fn transpose_f32(src: &[f32], w: usize, h: usize, channels: usize) -> Vec<f32> {
    const TILE: usize = 32;
    let mut dst = vec![0.0f32; w * h * channels];
    if w == 0 || h == 0 {
        return dst;
    }

    // Partition output rows (x in 0..w) in TILE-sized bands so every Rayon worker writes disjoint memory
    let out_row_stride = h * channels;
    dst.par_chunks_mut(TILE * out_row_stride).enumerate().for_each(|(tx, out_band)| {
        let x0 = tx * TILE;
        let x1 = (x0 + TILE).min(w);
        let tile_w = x1 - x0;
        for y0 in (0..h).step_by(TILE) {
            let y1 = (y0 + TILE).min(h);
            if channels == 1 {
                for y in y0..y1 {
                    let src_row = &src[y * w + x0..y * w + x1];
                    for (dx, &v) in src_row.iter().enumerate() {
                        out_band[dx * h + y] = v;
                    }
                }
            } else {
                for y in y0..y1 {
                    let src_row = &src[(y * w + x0) * channels..(y * w + x1) * channels];
                    for dx in 0..tile_w {
                        let s = dx * channels;
                        let d = (dx * h + y) * channels;
                        out_band[d..d + channels].copy_from_slice(&src_row[s..s + channels]);
                    }
                }
            }
        }
    });
    dst
}

/// One horizontal running-sum box pass over contiguous rows of length `line_len` (each pixel has `channels` floats).
fn box_rows(src: &[f32], dst: &mut [f32], line_len: usize, channels: usize, radius: i64, tile: bool) {
    let inverse_width = 1.0 / (2 * radius + 1) as f64;
    let n = line_len as i64;
    let row_stride = line_len * channels;
    let r = radius as usize;
    let interior_fast = line_len > 2 * r + 1;

    dst.par_chunks_mut(row_stride).enumerate().for_each(|(line, out_row)| {
        let in_row = &src[line * row_stride..(line + 1) * row_stride];
        let sample = |i: i64, ch: usize| -> f64 {
            let idx = if tile { i.rem_euclid(n) } else { i.clamp(0, n - 1) } as usize;
            f64::from(in_row[idx * channels + ch])
        };

        for ch in 0..channels {
            let mut sum: f64 = (-radius..=radius).map(|i| sample(i, ch)).sum();
            if interior_fast {
                // Left border
                for i in 0..r {
                    out_row[i * channels + ch] = (sum * inverse_width) as f32;
                    sum += f64::from(in_row[(i + r + 1) * channels + ch]) - sample(i as i64 - radius, ch);
                }
                // Interior (zero wrap/clamp modulo)
                let mid_end = line_len - r - 1;
                for i in r..mid_end {
                    out_row[i * channels + ch] = (sum * inverse_width) as f32;
                    sum += f64::from(in_row[(i + r + 1) * channels + ch]) - f64::from(in_row[(i - r) * channels + ch]);
                }
                // Right border
                for i in mid_end..line_len {
                    out_row[i * channels + ch] = (sum * inverse_width) as f32;
                    sum += sample(i as i64 + radius + 1, ch) - f64::from(in_row[(i - r) * channels + ch]);
                }
            } else {
                for i in 0..line_len {
                    out_row[i * channels + ch] = (sum * inverse_width) as f32;
                    sum += sample(i as i64 + radius + 1, ch) - sample(i as i64 - radius, ch);
                }
            }
        }
    });
}

/// One running-sum box pass of radius `radius`.
fn box_1d(src: &Image, radius: i64, horizontal: bool, tile: bool) -> Image {
    if radius <= 0 {
        return src.clone();
    }

    let (w, h) = (src.width() as usize, src.height() as usize);
    let channels = src.channels();
    let data = if horizontal {
        let mut out = vec![0.0f32; w * h * channels];
        box_rows(src.data(), &mut out, w, channels, radius, tile);
        out
    } else {
        let transposed = transpose_f32(src.data(), w, h, channels);
        let mut blurred = vec![0.0f32; w * h * channels];
        box_rows(&transposed, &mut blurred, h, channels, radius, tile);
        drop(transposed);
        transpose_f32(&blurred, h, w, channels)
    };

    // A valid source has matching dimensions and data length
    Image::from_data(src.width(), src.height(), src.kind(), data, &limits_for(src)).unwrap_or_else(|_| src.clone())
}

/// Squared Euclidean distance (pixels) from every pixel to the nearest pixel where `inside` is
/// true (Felzenszwalb–Huttenlocher, exact, linear time). With `tile`, distances wrap.
pub fn distance_sq(w: u32, h: u32, inside: impl Fn(usize) -> bool + Sync, tile: bool) -> Vec<f32> {
    distance_transform::<false>(w, h, inside, tile).0
}

/// The same exact distance transform, also retaining the nearest mask pixel's row-major index.
/// Empty masks have no nearest pixel: their indices are `usize::MAX`.
pub fn distance_sq_with_sites(w: u32, h: u32, inside: impl Fn(usize) -> bool + Sync, tile: bool) -> (Vec<f32>, Vec<usize>) {
    distance_transform::<true>(w, h, inside, tile)
}

const DISTANCE_INF: f32 = 1e20;

struct EdtScratch {
    f: Vec<f64>,
    v: Vec<usize>,
    z: Vec<f64>,
}

impl EdtScratch {
    fn new(cap: usize) -> Self {
        Self { f: Vec::with_capacity(cap), v: vec![0; cap], z: vec![0.0; cap + 1] }
    }
}

fn edt_1d_into<const SITES: bool>(scratch: &mut EdtScratch, n: usize, out_offset: usize, out_d: &mut [f32], out_sites: &mut [usize]) {
    if n == 0 {
        return;
    }

    if scratch.v.len() < n {
        scratch.v.resize(n, 0);
    }
    if scratch.z.len() < n + 1 {
        scratch.z.resize(n + 1, 0.0);
    }
    let f = &scratch.f;
    let v = &mut scratch.v;
    let z = &mut scratch.z;

    let mut k = 0usize;
    v[0] = 0;
    z[0] = -(DISTANCE_INF as f64);
    z[1] = DISTANCE_INF as f64;
    for q in 1..n {
        let fq_q2 = f[q] + (q * q) as f64;
        let q2 = 2.0 * q as f64;
        let mut p = v[k];
        let mut s = (fq_q2 - (f[p] + (p * p) as f64)) / (q2 - 2.0 * p as f64);
        while k > 0 && s <= z[k] {
            k -= 1;
            p = v[k];
            s = (fq_q2 - (f[p] + (p * p) as f64)) / (q2 - 2.0 * p as f64);
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = DISTANCE_INF as f64;
    }

    k = 0;
    let out_len = out_d.len();
    for (idx, dq) in out_d.iter_mut().enumerate() {
        let q = out_offset + idx;
        let qf = q as f64;
        while k + 1 < n && z[k + 1] < qf {
            k += 1;
        }
        let p = v[k];
        let diff = qf - p as f64;
        *dq = (diff * diff + f[p]) as f32;
        if SITES {
            out_sites[idx] = if f[p] < DISTANCE_INF as f64 { p } else { usize::MAX };
        }
    }
    let _ = out_len;
}

fn distance_transform<const SITES: bool>(w: u32, h: u32, inside: impl Fn(usize) -> bool + Sync, tile: bool) -> (Vec<f32>, Vec<usize>) {
    let (w, h) = (w as usize, h as usize);
    if w == 0 || h == 0 {
        return (Vec::new(), Vec::new());
    }

    // Tiling: run on a 3x tiled line so distances cross the edges
    let rep = if tile { 3 } else { 1 };

    // Columns (stored column-major in flat buffers to avoid per-column heap allocations)
    let mut col_d = vec![0.0f32; w * h];
    let mut col_s = if SITES { vec![usize::MAX; w * h] } else { Vec::new() };
    let col_off = if tile { h } else { 0 };
    let col_n = h * rep;
    if SITES {
        col_d.par_chunks_mut(h).zip(col_s.par_chunks_mut(h)).enumerate().for_each_init(
            || EdtScratch::new(col_n),
            |scratch, (x, (cd, cs))| {
                scratch.f.clear();
                for y in 0..h {
                    scratch.f.push(if inside(y * w + x) { 0.0 } else { f64::from(DISTANCE_INF) });
                }
                if tile {
                    scratch.f.extend_from_within(0..h);
                    scratch.f.extend_from_within(0..h);
                }
                edt_1d_into::<true>(scratch, col_n, col_off, cd, cs);
            },
        );
    } else {
        col_d.par_chunks_mut(h).enumerate().for_each_init(
            || EdtScratch::new(col_n),
            |scratch, (x, cd)| {
                scratch.f.clear();
                for y in 0..h {
                    scratch.f.push(if inside(y * w + x) { 0.0 } else { f64::from(DISTANCE_INF) });
                }
                if tile {
                    scratch.f.extend_from_within(0..h);
                    scratch.f.extend_from_within(0..h);
                }
                edt_1d_into::<false>(scratch, col_n, col_off, cd, &mut []);
            },
        );
    }

    // Transpose column-major distances into row-major so the row pass reads contiguous lines
    let row_in_d = transpose_f32(&col_d, h, w, 1);
    drop(col_d);

    let mut distances = vec![0.0f32; w * h];
    let mut sites = if SITES { vec![usize::MAX; w * h] } else { Vec::new() };
    let row_off = if tile { w } else { 0 };
    let row_n = w * rep;
    if SITES {
        distances.par_chunks_mut(w).zip(sites.par_chunks_mut(w)).enumerate().for_each_init(
            || (EdtScratch::new(row_n), vec![usize::MAX; w]),
            |(scratch, x_sites), (y, (rd, rs))| {
                scratch.f.clear();
                let row = &row_in_d[y * w..(y + 1) * w];
                for &val in row {
                    scratch.f.push(f64::from(val));
                }
                if tile {
                    scratch.f.extend_from_within(0..w);
                    scratch.f.extend_from_within(0..w);
                }
                edt_1d_into::<true>(scratch, row_n, row_off, rd, x_sites);
                for (dst, &sx) in rs.iter_mut().zip(x_sites.iter()) {
                    *dst = if sx == usize::MAX {
                        usize::MAX
                    } else {
                        let x = sx % w;
                        let sy = col_s[x * h + y];
                        if sy == usize::MAX { usize::MAX } else { (sy % h) * w + x }
                    };
                }
            },
        );
    } else {
        distances.par_chunks_mut(w).enumerate().for_each_init(
            || EdtScratch::new(row_n),
            |scratch, (y, rd)| {
                scratch.f.clear();
                let row = &row_in_d[y * w..(y + 1) * w];
                for &val in row {
                    scratch.f.push(f64::from(val));
                }
                if tile {
                    scratch.f.extend_from_within(0..w);
                    scratch.f.extend_from_within(0..w);
                }
                edt_1d_into::<false>(scratch, row_n, row_off, rd, &mut []);
            },
        );
    }

    (distances, sites)
}

/// Slope (dx, dy) of a gray image by central differences, in value units per pixel.
#[inline]
pub fn slope(img: &Image, x: i64, y: i64, tile: bool) -> (f32, f32) {
    let dx = (img.get_gray(x + 1, y, tile) - img.get_gray(x - 1, y, tile)) * 0.5;
    let dy = (img.get_gray(x, y + 1, tile) - img.get_gray(x, y - 1, tile)) * 0.5;
    (dx, dy)
}

pub fn rgb_to_hsl(c: [f32; 4]) -> [f32; 3] {
    let (r, g, b) = (c[0], c[1], c[2]);
    let mx = r.max(g).max(b);
    let mn = r.min(g).min(b);
    let l = (mx + mn) * 0.5;
    if (mx - mn).abs() < 1e-7 {
        return [0.0, 0.0, l];
    }

    let d = mx - mn;
    let s = if l > 0.5 { d / (2.0 - mx - mn) } else { d / (mx + mn) };
    let h = if mx == r {
        (g - b) / d + if g < b { 6.0 } else { 0.0 }
    } else if mx == g {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };

    [h / 6.0, s, l]
}

pub fn hsl_to_rgb(h: f32, s: f32, l: f32) -> [f32; 3] {
    if s <= 0.0 {
        return [l, l, l];
    }

    let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
    let p = 2.0 * l - q;
    let f = |mut t: f32| {
        t = t.rem_euclid(1.0);
        if t < 1.0 / 6.0 {
            p + (q - p) * 6.0 * t
        } else if t < 0.5 {
            q
        } else if t < 2.0 / 3.0 {
            p + (q - p) * (2.0 / 3.0 - t) * 6.0
        } else {
            p
        }
    };
    [f(h + 1.0 / 3.0), f(h), f(h - 1.0 / 3.0)]
}

pub const BLEND_MODES: &[&str] = &[
    "Copy",
    "Add (Linear dodge)",
    "Subtract",
    "Multiply",
    "Add Sub",
    "Max (Lighten)",
    "Min (Darken)",
    "Switch",
    "Divide",
    "Overlay",
    "Screen",
    "Soft light",
    "Difference",
    "Color dodge",
    "Color burn",
    "Hard light",
];

/// Blend one channel: `b` background, `f` foreground.
#[inline]
pub fn blend_channel(mode: i32, b: f32, f: f32) -> f32 {
    match mode {
        1 => b + f,
        2 => b - f,
        3 => b * f,
        4 => b + 2.0 * f - 1.0,
        5 => b.max(f),
        6 => b.min(f),
        8 => {
            if f.abs() < 1e-6 {
                if b > 0.0 { 1.0 } else { 0.0 }
            } else {
                b / f
            }
        }
        9 => {
            if b < 0.5 {
                2.0 * b * f
            } else {
                1.0 - 2.0 * (1.0 - b) * (1.0 - f)
            }
        }
        10 => 1.0 - (1.0 - b) * (1.0 - f),
        11 => {
            // W3C soft light
            if f <= 0.5 {
                b - (1.0 - 2.0 * f) * b * (1.0 - b)
            } else {
                let d = if b <= 0.25 { ((16.0 * b - 12.0) * b + 4.0) * b } else { b.max(0.0).sqrt() };
                b + (2.0 * f - 1.0) * (d - b)
            }
        }
        12 => (b - f).abs(),
        13 => {
            // W3C color dodge: a black background stays black, even under a white foreground
            if f >= 1.0 { if b > 0.0 { 1.0 } else { 0.0 } } else { (b / (1.0 - f)).min(1.0) }
        }
        14 => {
            // W3C color burn: a white background stays white, even under a black foreground
            if f <= 0.0 { if b < 1.0 { 0.0 } else { 1.0 } } else { 1.0 - ((1.0 - b) / f).min(1.0) }
        }
        15 => {
            if f < 0.5 {
                2.0 * b * f
            } else {
                1.0 - 2.0 * (1.0 - b) * (1.0 - f)
            }
        }
        _ => f,
    }
}

/// Blend `fg` over `bg` with `mode`, weighted by `opacity` (0..1, already including any mask).
/// `blend_alpha`: composite coverage; false preserves background alpha.
#[inline]
pub fn blend_px(mode: i32, bg: [f32; 4], fg: [f32; 4], opacity: f32, color_blending: i32, blend_alpha: bool, clamp: bool) -> [f32; 4] {
    // Switch already interpolates RGBA; another alpha weight composites twice
    let t = opacity * if mode != 7 { fg[3] } else { 1.0 };
    let coverage = bg[3] * (1.0 - t) + t;
    let mut o = [0.0; 4];
    for k in 0..3 {
        let v = blend_channel(mode, bg[k], fg[k]);

        // Weight before adding to preserve integer-boundary rounding
        o[k] = match color_blending {
            1 if mode != 7 => bg[k] * (1.0 - opacity) + v * opacity,
            2 if mode != 7 && coverage > 0.0 => {
                let v = if clamp { v.clamp(0.0, 1.0) } else { v };
                (bg[k] * bg[3] * (1.0 - t) + fg[k] * t * (1.0 - bg[3]) + v * bg[3] * t) / coverage
            }
            2 if mode != 7 => bg[k],
            3 if mode == 0 => bg[k] * (1.0 - t) + fg[k] * opacity,
            _ => bg[k] * (1.0 - t) + v * t,
        };
        if clamp {
            o[k] = o[k].clamp(0.0, 1.0);
        }
    }

    o[3] = if mode == 7 {
        bg[3] * (1.0 - opacity) + fg[3] * opacity
    } else if blend_alpha || color_blending == 2 || (color_blending == 3 && mode == 0) {
        coverage
    } else {
        bg[3]
    };

    o
}

/// A gray image of the output size from per-pixel values, or an error when the size is out of
/// range or `v` does not hold one value per pixel.
pub fn gray_from(w: u32, h: u32, v: Vec<f32>) -> Result<Image, String> {
    Image::from_data(w, h, Kind::Gray, v, &Limits::default()).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_dodge_and_burn_keep_black_and_white_backgrounds() {
        const DODGE: i32 = 13;
        const BURN: i32 = 14;

        // (background, foreground, expected), W3C compositing formulas
        for (b, f, expected) in [(0.0, 1.0, 0.0), (0.0, 0.5, 0.0), (0.5, 1.0, 1.0), (0.25, 0.5, 0.5), (1.0, 0.75, 1.0)] {
            assert_eq!(blend_channel(DODGE, b, f), expected, "dodge b={b} f={f}");
        }

        for (b, f, expected) in [(1.0, 0.0, 1.0), (1.0, 0.5, 1.0), (0.5, 0.0, 0.0), (0.75, 0.5, 0.5), (0.0, 0.25, 0.0)] {
            assert_eq!(blend_channel(BURN, b, f), expected, "burn b={b} f={f}");
        }
    }

    #[test]
    fn blur_keeps_mean_and_tiles() {
        let img = Image::gray(32, 32, |x, y| if (x / 4 + y / 4) % 2 == 0 { 1.0 } else { 0.0 });
        for s in [1.0, 6.0] {
            let b = gaussian(&img, s, true);

            assert!((b.mean()[0] - img.mean()[0]).abs() < 1e-3, "sigma {s}");
        }
        let flat = gaussian(&Image::filled(8, 8, Kind::Color, [0.2, 0.4, 0.6, 1.0]), 4.0, false);

        assert!((flat.get(3, 3, false)[1] - 0.4).abs() < 1e-5);
    }

    #[test]
    fn box_blur_of_large_hdr_values_stays_finite() {
        let value = 0.75 * f32::MAX;
        let blurred = gaussian(&Image::filled(16, 4, Kind::Gray, [value; 4]), 5.0, true);

        for x in 0..16 {
            let v = blurred.get(x, 1, false)[0];

            assert!(v.is_finite() && (v / value - 1.0).abs() < 1e-5, "pixel {x} is {v}");
        }
    }

    #[test]
    fn blur_keeps_the_size_and_result_of_an_image_over_the_default_limits() {
        let w = craft_texture::MAX_SIDE + 1;
        let mut limits = Limits::default();
        limits.max_side = w;
        let values: Vec<f32> = (0..w).map(|x| if (x / 8) % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let Ok(img) = Image::from_data(w, 1, Kind::Gray, values, &limits) else {
            panic!("a {w}x1 image fits the raised limits");
        };

        for sigma in [1.0, 5.0] {
            let blurred = gaussian(&img, sigma, false);

            assert_eq!((blurred.width(), blurred.height()), (w, 1), "sigma {sigma}");
            assert_ne!(blurred.data(), img.data(), "sigma {sigma} left the pixels unchanged");
        }
    }

    #[test]
    fn gray_from_rejects_an_empty_size_and_a_wrong_value_count() {
        assert!(gray_from(0, 4, Vec::new()).is_err());
        assert!(gray_from(2, 2, vec![0.0; 3]).is_err());
        assert_eq!(gray_from(2, 2, vec![0.5; 4]).map(|image| image.mean()[0]), Ok(0.5));
    }

    #[test]
    fn edt_exact() {
        let d = distance_sq(9, 1, |i| i == 4, false);
        assert_eq!(d, vec![16.0, 9.0, 4.0, 1.0, 0.0, 1.0, 4.0, 9.0, 16.0]);
        let t = distance_sq(9, 1, |i| i == 0, true);
        assert_eq!(t[8], 1.0);
    }

    #[test]
    fn edt_sites_match_brute_force() {
        // Exhaust small masks; use deterministic masks and per-site checks for larger ones
        for (w, h) in [(1usize, 1usize), (1, 7), (7, 1), (2, 3), (3, 3), (5, 7), (8, 6)] {
            let cases = if w * h <= 9 { 1usize << (w * h) } else { 128 };
            for case in 0..cases {
                let mut state = case as u32 + 1;
                let mask: Vec<bool> = (0..w * h)
                    .map(|i| {
                        if w * h <= 9 {
                            case & (1 << i) != 0
                        } else {
                            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                            state.is_multiple_of(7)
                        }
                    })
                    .collect();
                for tile in [false, true] {
                    let (d, sites) = distance_sq_with_sites(w as u32, h as u32, |i| mask[i], tile);

                    assert_eq!(d, distance_sq(w as u32, h as u32, |i| mask[i], tile));

                    for i in 0..w * h {
                        let distance = |site: usize| {
                            let dx = (i % w).abs_diff(site % w);
                            let dy = (i / w).abs_diff(site / w);
                            let (dx, dy) = if tile { (dx.min(w - dx), dy.min(h - dy)) } else { (dx, dy) };
                            (dx * dx + dy * dy) as f32
                        };
                        let expected = (0..mask.len()).filter(|&site| mask[site]).map(distance).reduce(f32::min);
                        if let Some(expected) = expected {
                            assert_eq!(d[i], expected, "{w}x{h}, mask {case}, tile {tile}, pixel {i}");
                            assert!(sites[i] < mask.len() && mask[sites[i]]);
                            assert_eq!(distance(sites[i]), expected);
                        } else {
                            assert_eq!(sites[i], usize::MAX);
                            assert!(d[i] >= DISTANCE_INF);
                        }
                    }
                }
            }
        }

        assert!(distance_sq_with_sites(0, 3, |_| true, true).0.is_empty());
        assert!(distance_sq_with_sites(3, 0, |_| true, false).1.is_empty());
    }

    #[test]
    fn hsl_round_trip() {
        let c = [0.8, 0.3, 0.1, 1.0];
        let h = rgb_to_hsl(c);
        let r = hsl_to_rgb(h[0], h[1], h[2]);
        for k in 0..3 {
            assert!((r[k] - c[k]).abs() < 1e-5);
        }
    }
}
