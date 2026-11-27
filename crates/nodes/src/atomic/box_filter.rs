use craft_texture::{Image, Limits, MAX_SIDE};
use rayon::prelude::*;

pub(super) fn axis(input: &Image, radius: f32, horizontal: bool, tile: bool) -> Image {
    if !radius.is_finite() || radius <= 0.5 {
        return input.clone();
    }

    // Bound work for hostile strengths to the largest supported image dimension
    let radius = radius.min(MAX_SIDE as f32);
    let extent = (radius + 0.5).floor() as i64;
    let weight = 1.0 / (2.0 * f64::from(radius));
    let edge_weight = (0.5 - (extent as f64 - 0.5) * weight).max(0.0);
    let (w, h) = (input.width() as usize, input.height() as usize);
    let channels = input.channels();

    let data = if horizontal {
        let mut out = vec![0.0; w * h * channels];
        filter_rows(input.data(), &mut out, w, channels, extent, weight, edge_weight, tile);
        out
    } else {
        let transposed = crate::ops::transpose_f32(input.data(), w, h, channels);
        let mut blurred = vec![0.0; w * h * channels];
        filter_rows(&transposed, &mut blurred, h, channels, extent, weight, edge_weight, tile);
        drop(transposed);
        crate::ops::transpose_f32(&blurred, h, w, channels)
    };

    let mut limits = Limits::default();
    limits.max_side = u32::MAX;
    limits.max_pixels = u64::MAX;
    limits.max_bytes = u64::MAX;

    // `data` keeps the size and kind of a valid image, so `from_data` accepts it
    Image::from_data(input.width(), input.height(), input.kind(), data, &limits).unwrap_or_else(|_| input.clone())
}

#[allow(clippy::too_many_arguments)]
fn filter_rows(src: &[f32], dst: &mut [f32], length: usize, channels: usize, extent: i64, weight: f64, edge_weight: f64, tile: bool) {
    let n = length as i64;
    let row_stride = length * channels;
    dst.par_chunks_mut(row_stride).enumerate().for_each(|(line, output)| {
        let in_row = &src[line * row_stride..(line + 1) * row_stride];
        let get = |position: i64, channel: usize| -> f32 {
            let idx = if tile { position.rem_euclid(n) } else { position.clamp(0, n - 1) } as usize;
            in_row[idx * channels + channel]
        };
        for channel in 0..channels {
            // Count NaN, positive infinity and negative infinity separately
            let mut sum = 0.0_f64;
            let mut non_finite = [0_i64; 3];
            for position in (1 - extent)..extent {
                accumulate(&mut sum, &mut non_finite, get(position, channel), 1);
            }

            for (position, pixel) in output.chunks_exact_mut(channels).enumerate() {
                let position = position as i64;
                let mut value = window_total(sum, non_finite) * weight;
                if edge_weight > 0.0 {
                    value += (f64::from(get(position - extent, channel)) + f64::from(get(position + extent, channel))) * edge_weight;
                }

                pixel[channel] = value as f32;
                accumulate(&mut sum, &mut non_finite, get(position + extent, channel), 1);
                accumulate(&mut sum, &mut non_finite, get(position - extent + 1, channel), -1);
            }
        }
    });
}

fn accumulate(sum: &mut f64, non_finite: &mut [i64; 3], value: f32, sign: i64) {
    if value.is_finite() {
        *sum += f64::from(value) * sign as f64;
    } else {
        let index = if value.is_nan() {
            0
        } else if value.is_sign_positive() {
            1
        } else {
            2
        };
        non_finite[index] += sign;
    }
}

fn window_total(sum: f64, non_finite: [i64; 3]) -> f64 {
    // A non-finite sample must stop affecting the running sum once it leaves the window
    if non_finite[0] > 0 || (non_finite[1] > 0 && non_finite[2] > 0) {
        f64::NAN
    } else if non_finite[1] > 0 {
        f64::INFINITY
    } else if non_finite[2] > 0 {
        f64::NEG_INFINITY
    } else {
        sum
    }
}
