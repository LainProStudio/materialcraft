use super::ATOMIC;
use crate::{CookContext, CookResult, NodeDefinition, node};
use craft_texture::{Image, Kind, Limits};
use rayon::prelude::*;

pub(super) fn definition() -> NodeDefinition {
    node("levels", "Levels", ATOMIC)
        .desc("Remaps the input range, midtones and output range.")
        .atomic()
        .any_in("input", "Input")
        .float4("in_low", "Level in low", [0.0; 4], 0.0, 1.0)
        .float4("in_mid", "Level in mid", [0.5; 4], 0.0, 1.0)
        .float4("in_high", "Level in high", [1.0; 4], 0.0, 1.0)
        .float4("out_low", "Level out low", [0.0; 4], 0.0, 1.0)
        .float4("out_high", "Level out high", [1.0; 4], 0.0, 1.0)
        .any_out()
        .cook(levels)
}

#[derive(Clone, Copy)]
struct ChannelLevels {
    in_low: f32,
    in_high: f32,
    span: f32,
    collapsed: bool,
    gamma: f32,
    unit_gamma: bool,
    out_low: f32,
    out_span: f32,
}

impl ChannelLevels {
    fn new(in_low: f32, in_mid: f32, in_high: f32, out_low: f32, out_high: f32) -> Self {
        let span = in_high - in_low;
        let collapsed = span.abs() < 1e-6;
        let mid = if in_mid.is_finite() { in_mid.clamp(0.0, 1.0) } else { 0.5 };
        let gamma = if mid <= 0.5 { 1.0 / (9.0 - 16.0 * mid) } else { 16.0 * mid - 7.0 };
        Self { in_low, in_high, span, collapsed, gamma, unit_gamma: (gamma - 1.0).abs() < 1e-7, out_low, out_span: out_high - out_low }
    }

    #[inline(always)]
    fn apply(&self, v: f32) -> f32 {
        let t = if self.collapsed { if v >= self.in_high { 1.0 } else { 0.0 } } else { ((v - self.in_low) / self.span).clamp(0.0, 1.0) };
        let t = if t.is_finite() { t } else { 0.0 };
        let curved = if self.unit_gamma { t } else { t.powf(self.gamma) };
        self.out_low + curved * self.out_span
    }
}

fn levels(context: &CookContext) -> CookResult {
    let image = context.any_or(0, Kind::Gray, [0.0; 4]);
    let low = context.v4("in_low");
    let mid = context.v4("in_mid");
    let high = context.v4("in_high");
    let output_low = context.v4("out_low");
    let output_high = context.v4("out_high");
    let ch: [ChannelLevels; 4] = std::array::from_fn(|c| ChannelLevels::new(low[c], mid[c], high[c], output_low[c], output_high[c]));

    let (w, h) = (image.width(), image.height());
    let src = image.data();
    let mut out = vec![0.0f32; src.len()];
    if image.is_gray() {
        let c0 = ch[0];
        out.par_chunks_mut(w as usize).zip(src.par_chunks(w as usize)).for_each(|(dst_row, src_row)| {
            for (d, &s) in dst_row.iter_mut().zip(src_row) {
                *d = c0.apply(s);
            }
        });
    } else {
        let row_floats = w as usize * 4;
        out.par_chunks_mut(row_floats).zip(src.par_chunks(row_floats)).for_each(|(dst_row, src_row)| {
            for (d, s) in dst_row.as_chunks_mut::<4>().0.iter_mut().zip(src_row.as_chunks::<4>().0) {
                d[0] = ch[0].apply(s[0]);
                d[1] = ch[1].apply(s[1]);
                d[2] = ch[2].apply(s[2]);
                d[3] = ch[3].apply(s[3]);
            }
        });
    }

    Ok(vec![Image::from_data(w, h, image.kind(), out, &Limits::default()).map_err(|e| e.to_string())?])
}

pub fn levels_fn(v: f32, in_low: f32, in_mid: f32, in_high: f32, out_low: f32, out_high: f32) -> f32 {
    let span = in_high - in_low;
    let t = if span.abs() < 1e-6 { if v >= in_high { 1.0 } else { 0.0 } } else { ((v - in_low) / span).clamp(0.0, 1.0) };

    // NaN (from an input or a hostile parameter) would otherwise spread through the graph
    let t = if t.is_finite() { t } else { 0.0 };
    let mid = if in_mid.is_finite() { in_mid.clamp(0.0, 1.0) } else { 0.5 };

    // The midpoint slider controls reciprocal gamma below neutral and gamma above it
    // Treating it as an input value that maps to 0.5 changes authored levels curves
    let g = if mid <= 0.5 { 1.0 / (9.0 - 16.0 * mid) } else { 16.0 * mid - 7.0 };
    out_low + t.powf(g) * (out_high - out_low)
}

#[cfg(test)]
mod levels_tests {
    use super::levels_fn;

    #[test]
    fn levels_remaps_the_input_range_midtones_and_output_range() {
        let close = |a: f32, b: f32| (a - b).abs() < 1e-5;

        // Identity by default
        for v in [0.0, 0.25, 0.5, 1.0] {
            assert!(close(levels_fn(v, 0.0, 0.5, 1.0, 0.0, 1.0), v));
        }

        // The input range stretches and clips
        assert!(close(levels_fn(0.4, 0.2, 0.5, 0.6, 0.0, 1.0), 0.5));
        assert!(close(levels_fn(0.1, 0.2, 0.5, 0.6, 0.0, 1.0), 0.0));
        assert!(close(levels_fn(0.9, 0.2, 0.5, 0.6, 0.0, 1.0), 1.0));

        // The same gamma curve applies after normalizing the input range
        assert!(close(levels_fn(0.25, 0.0, 0.25, 1.0, 0.0, 1.0), 0.7578583));
        assert!(close(levels_fn(0.3, 0.2, 0.25, 0.6, 0.0, 1.0), 0.7578583));

        // The output range, inverted when low > high
        assert!(close(levels_fn(0.5, 0.0, 0.5, 1.0, 0.2, 0.6), 0.4));
        assert!(close(levels_fn(0.0, 0.0, 0.5, 1.0, 1.0, 0.0), 1.0));

        // A collapsed input range is a threshold; NaN never escapes
        assert_eq!(levels_fn(0.49, 0.5, 0.5, 0.5, 0.0, 1.0), 0.0);
        assert_eq!(levels_fn(0.5, 0.5, 0.5, 0.5, 0.0, 1.0), 1.0);
        assert!(levels_fn(f32::NAN, 0.0, 0.5, 1.0, 0.0, 1.0).is_finite());
        assert!(levels_fn(0.5, f32::NAN, 0.5, f32::NAN, 0.0, 1.0).is_finite());
    }
}
