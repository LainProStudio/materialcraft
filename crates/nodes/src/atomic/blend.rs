use super::ATOMIC;
use crate::ops;
use crate::ops::BLEND_MODES;
use crate::{CookContext, CookResult, NodeDefinition, node};
use craft_texture::{Image, Kind, Limits};
use rayon::prelude::*;

pub(super) fn definition() -> NodeDefinition {
    node("blend", "Blend", ATOMIC)
        .desc("Blends a foreground over a background with a blending mode, weighted by opacity and an optional mask.")
        .atomic()
        .any_in("foreground", "Foreground")
        .any_in("background", "Background")
        .primary_input("background")
        .gray_in("mask", "Opacity")
        .float("opacity", "Opacity", 1.0, 0.0, 1.0)
        .choice("mode", "Blending mode", BLEND_MODES, 0)
        .boolean("alpha", "Alpha blending", true)
        .choice("color_blending", "Alpha interpretation", &["Source alpha", "Ignore alpha", "Straight", "Premultiplied"], 0)
        .any_out()
        .cook(blend)
}

fn blend(context: &CookContext) -> CookResult {
    let (foreground, background) = (context.sized(0), context.sized(1));
    let kind = if [&foreground, &background].iter().any(|i| i.as_ref().is_some_and(|i| !i.is_gray())) { Kind::Color } else { Kind::Gray };
    let (w, h) = craft_texture::clamp_size(context.w, context.h);
    let foreground = foreground.unwrap_or_else(|| Image::filled(w, h, kind, [0.0, 0.0, 0.0, 0.0]));
    let background = background.unwrap_or_else(|| Image::filled(w, h, kind, [0.0, 0.0, 0.0, 1.0]));
    let mask = context.sized(2).map(|m| m.to_gray());
    let (mode, opacity) = (context.i("mode"), context.f("opacity"));
    let color_blending = context.i("color_blending");
    let blend_alpha = context.b("alpha") && !foreground.is_gray();
    let clamp =
        matches!(context.format, materialcraft_core::OutputFormat::U8 | materialcraft_core::OutputFormat::U16 | materialcraft_core::OutputFormat::Inherit);

    let (wi, hi) = (w as usize, h as usize);
    let mask_data = mask.as_ref().map(|m| m.data());

    if kind == Kind::Gray {
        // Grayscale inputs have implicit alpha = 1.0, so only 1 channel needs blending
        let fg = foreground.data();
        let bg = background.data();
        let mut out = vec![0.0f32; wi * hi];
        let premul_copy = color_blending == 3 && mode == 0;
        let straight_clamp = color_blending == 2 && mode != 7 && clamp;
        out.par_chunks_mut(wi).enumerate().for_each(|(y, out_row)| {
            let off = y * wi;
            let bg_row = &bg[off..off + wi];
            let fg_row = &fg[off..off + wi];
            let m_row = mask_data.map(|m| &m[off..off + wi]);
            for x in 0..wi {
                let t = opacity * m_row.map_or(1.0, |m| m[x]);
                let b = bg_row[x];
                let f = fg_row[x];
                let mut v = if premul_copy {
                    b * (1.0 - t) + f * t
                } else {
                    let blended = ops::blend_channel(mode, b, f);
                    let blended = if straight_clamp { blended.clamp(0.0, 1.0) } else { blended };
                    b * (1.0 - t) + blended * t
                };
                if clamp {
                    v = v.clamp(0.0, 1.0);
                }
                out_row[x] = v;
            }
        });
        return Ok(vec![Image::from_data(w, h, Kind::Gray, out, &Limits::default()).map_err(|e| e.to_string())?]);
    }

    // Color output: normalize inputs to 4 channels for direct slice access
    let fg_img = foreground.to_color();
    let bg_img = background.to_color();
    let fg = fg_img.data();
    let bg = bg_img.data();
    let mut out = vec![0.0f32; wi * hi * 4];
    out.par_chunks_mut(wi * 4).enumerate().for_each(|(y, out_row)| {
        let px_off = y * wi;
        let ch_off = px_off * 4;
        let bg_row = &bg[ch_off..ch_off + wi * 4];
        let fg_row = &fg[ch_off..ch_off + wi * 4];
        let m_row = mask_data.map(|m| &m[px_off..px_off + wi]);
        for (x, dst) in out_row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let i = x * 4;
            let b = [bg_row[i], bg_row[i + 1], bg_row[i + 2], bg_row[i + 3]];
            let f = [fg_row[i], fg_row[i + 1], fg_row[i + 2], fg_row[i + 3]];
            let m = m_row.map_or(1.0, |m| m[x]);
            *dst = ops::blend_px(mode, b, f, opacity * m, color_blending, blend_alpha, clamp);
        }
    });
    Ok(vec![Image::from_data(w, h, Kind::Color, out, &Limits::default()).map_err(|e| e.to_string())?])
}
