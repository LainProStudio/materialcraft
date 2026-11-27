use materialcraft_core::{PatternGraph, PatternNode, PatternNodeKind, Value};
use std::collections::BTreeMap;

use super::{
    catalog,
    fx_program::{FxOutput, FxProgram, FxRegs, FxVarSchema, FxVisitInputs},
    pattern_stamp::PatternStamp,
    raster,
};
use crate::function::{EvaluationContext, EvaluationState, Program, compile};
use crate::{CookContext, CookResult};

// A 512-by-512 quadtree needs 349,525 visits including internal quadrants
// Cap deeper trees; the rasterizer separately bounds pixel work
const MAX_VISITS: usize = 1_048_576;

#[derive(Clone, Copy)]
struct CompiledVisit {
    node: usize,
    center: [f32; 2],
    extent: f32,
    depth: u32,
    number: u32,
    seed: u32,
}

#[derive(Clone, Copy)]
enum ParamSlot {
    PatternType,
    Opacity,
    BranchOffset,
    FrameOffset,
    PatternSize,
    PatternRotation,
    PatternSuppl,
    BlendingMode,
    ImageFiltering,
    ImagePremul,
    ImageIndex,
    RandomSeed,
    RandomInherited,
    Switch,
    NumberAdded,
}

enum CompiledParamFn {
    Fast(FxProgram),
    Slow(Program),
}

struct DynamicParam {
    slot: ParamSlot,
    func: CompiledParamFn,
}

#[derive(Clone, Copy)]
struct QuadrantStatic {
    patterntype: i32,
    color: [f32; 4],
    scalar_color: bool,
    opacity_valid: bool,
    branchoffset: [f32; 2],
    frameoffset: [f32; 2],
    patternsize: [f32; 2],
    patternrotation: f32,
    patternsuppl: f32,
    blendingmode: i32,
    imagefiltering: i32,
    imagepremul: bool,
    imageindex: i32,
    randomseed: i32,
    randominherited: bool,
}

struct CompiledNode {
    kind: PatternNodeKind,
    children: [Option<usize>; 4],
    child_count: usize,
    pre_random_override: Option<u32>,
    quadrant: QuadrantStatic,
    switch_static: bool,
    number_added_static: i32,
    dynamic: Vec<DynamicParam>,
}

fn param_slot(name: &str) -> Option<ParamSlot> {
    match name {
        "patterntype" => Some(ParamSlot::PatternType),
        "opacity" => Some(ParamSlot::Opacity),
        "branchoffset" => Some(ParamSlot::BranchOffset),
        "frameoffset" => Some(ParamSlot::FrameOffset),
        "patternsize" => Some(ParamSlot::PatternSize),
        "patternrotation" => Some(ParamSlot::PatternRotation),
        "patternsuppl" => Some(ParamSlot::PatternSuppl),
        "blendingmode" => Some(ParamSlot::BlendingMode),
        "imagefiltering" => Some(ParamSlot::ImageFiltering),
        "imagepremul" => Some(ParamSlot::ImagePremul),
        "imageindex" => Some(ParamSlot::ImageIndex),
        "randomseed" => Some(ParamSlot::RandomSeed),
        "randominherited" => Some(ParamSlot::RandomInherited),
        "switch" => Some(ParamSlot::Switch),
        "numberadded" => Some(ParamSlot::NumberAdded),
        _ => None,
    }
}

fn static_f32(node: &PatternNode, defaults: &BTreeMap<&str, &Value>, name: &str) -> f32 {
    node.parameters.get(name).or_else(|| defaults.get(name).copied()).and_then(Value::as_f32).unwrap_or(0.0)
}

fn static_i32(node: &PatternNode, defaults: &BTreeMap<&str, &Value>, name: &str) -> i32 {
    node.parameters.get(name).or_else(|| defaults.get(name).copied()).and_then(Value::as_i32).unwrap_or(0)
}

fn static_bool(node: &PatternNode, defaults: &BTreeMap<&str, &Value>, name: &str) -> bool {
    node.parameters.get(name).or_else(|| defaults.get(name).copied()).and_then(Value::as_bool).unwrap_or(false)
}

fn static_vec2(node: &PatternNode, defaults: &BTreeMap<&str, &Value>, name: &str) -> [f32; 2] {
    node.parameters.get(name).or_else(|| defaults.get(name).copied()).and_then(Value::as_vec4).map(|v| [v[0], v[1]]).unwrap_or([0.0; 2])
}

fn compile_pattern_graph(graph: &PatternGraph, context: &CookContext) -> Result<(Vec<CompiledNode>, Option<usize>, FxVarSchema, bool), String> {
    let mut id_to_index = BTreeMap::new();
    for (idx, node) in graph.nodes.iter().enumerate() {
        id_to_index.insert(node.id, idx);
    }

    let mut raw_programs: Vec<Vec<(ParamSlot, Program)>> = Vec::with_capacity(graph.nodes.len());
    for node in &graph.nodes {
        let mut node_progs = Vec::new();
        if node.kind != PatternNodeKind::Dot {
            let definition = catalog::definition(node.kind).ok_or("missing pattern definition")?;
            for parameter in &definition.params {
                if let Some(Value::Function(function)) = node.parameters.get(parameter.id) {
                    let program = compile(function, context.seed)?;
                    if let Some(slot) = param_slot(parameter.id) {
                        node_progs.push((slot, program));
                    }
                }
            }
        }
        raw_programs.push(node_progs);
    }

    let schema = FxVarSchema::collect(raw_programs.iter().flat_map(|ps| ps.iter().map(|(_, p)| p)), context.vars);
    let size = [context.w as f32, context.h as f32];

    // Compile every parameter function against the shared variable schema; if any parameter falls
    // back to the slow interpreter while shared variables exist, run all parameters through the
    // slow interpreter so both see one consistent `BTreeMap` of variables.
    let mut fast_plans: Vec<Vec<Option<FxProgram>>> = Vec::with_capacity(raw_programs.len());
    let mut needs_slow_fallback = false;
    for node_progs in &raw_programs {
        let mut plans = Vec::with_capacity(node_progs.len());
        for (_, program) in node_progs {
            let plan = FxProgram::try_compile(program, context.vars, size, context.tile, &schema);
            if plan.is_none() {
                needs_slow_fallback = true;
            }
            plans.push(plan);
        }
        fast_plans.push(plans);
    }
    let allow_fast = !(needs_slow_fallback && !schema.is_empty());

    let mut compiled = Vec::with_capacity(graph.nodes.len());

    for ((node, node_progs), plans) in graph.nodes.iter().zip(raw_programs).zip(fast_plans) {
        let mut children = [None; 4];
        let child_count = node.children.len().min(4);
        for (slot, child) in node.children.iter().take(4).enumerate() {
            children[slot] = match child {
                Some(child_id) => Some(*id_to_index.get(child_id).ok_or("missing pattern node")?),
                None => None,
            };
        }

        if node.kind == PatternNodeKind::Dot {
            compiled.push(CompiledNode {
                kind: PatternNodeKind::Dot,
                children,
                child_count,
                pre_random_override: None,
                quadrant: QuadrantStatic {
                    patterntype: 0,
                    color: [1.0; 4],
                    scalar_color: false,
                    opacity_valid: true,
                    branchoffset: [0.0; 2],
                    frameoffset: [0.0; 2],
                    patternsize: [1.0; 2],
                    patternrotation: 0.0,
                    patternsuppl: 0.0,
                    blendingmode: 0,
                    imagefiltering: 1,
                    imagepremul: false,
                    imageindex: 1,
                    randomseed: 0,
                    randominherited: true,
                },
                switch_static: false,
                number_added_static: 1,
                dynamic: Vec::new(),
            });
            continue;
        }

        let definition = catalog::definition(node.kind).ok_or("missing pattern definition")?;
        let defaults: BTreeMap<&str, &Value> = definition.params.iter().map(|p| (p.id, &p.default)).collect();

        let pre_random_override = if node.parameters.get("randominherited").and_then(Value::as_bool) == Some(false) {
            Some(node.parameters.get("randomseed").and_then(Value::as_i32).unwrap_or(0) as u32)
        } else {
            None
        };

        let opacity_val = node.parameters.get("opacity").or_else(|| defaults.get("opacity").copied());
        let (color, scalar_color, opacity_valid) = match opacity_val {
            Some(Value::Float(v)) => ([*v, *v, *v, 1.0], true, true),
            Some(Value::Function(_)) => ([1.0; 4], false, true),
            Some(v) => match v.as_vec4() {
                Some(c) => (c, false, true),
                None => ([1.0; 4], false, false),
            },
            None => ([1.0; 4], false, true),
        };

        let quadrant = QuadrantStatic {
            patterntype: static_i32(node, &defaults, "patterntype"),
            color,
            scalar_color,
            opacity_valid,
            branchoffset: static_vec2(node, &defaults, "branchoffset"),
            frameoffset: static_vec2(node, &defaults, "frameoffset"),
            patternsize: static_vec2(node, &defaults, "patternsize"),
            patternrotation: static_f32(node, &defaults, "patternrotation"),
            patternsuppl: static_f32(node, &defaults, "patternsuppl"),
            blendingmode: static_i32(node, &defaults, "blendingmode"),
            imagefiltering: static_i32(node, &defaults, "imagefiltering"),
            imagepremul: static_bool(node, &defaults, "imagepremul"),
            imageindex: static_i32(node, &defaults, "imageindex"),
            randomseed: static_i32(node, &defaults, "randomseed"),
            randominherited: static_bool(node, &defaults, "randominherited"),
        };

        let switch_static = static_bool(node, &defaults, "switch");
        let number_added_static = static_i32(node, &defaults, "numberadded");

        let mut dynamic = Vec::with_capacity(node_progs.len());
        for ((slot, program), fast_plan) in node_progs.into_iter().zip(plans) {
            let compiled_fn = match fast_plan {
                Some(fast) if allow_fast => CompiledParamFn::Fast(fast),
                _ => CompiledParamFn::Slow(program),
            };
            dynamic.push(DynamicParam { slot, func: compiled_fn });
        }

        compiled.push(CompiledNode { kind: node.kind, children, child_count, pre_random_override, quadrant, switch_static, number_added_static, dynamic });
    }

    let root = match graph.root {
        Some(root_id) => Some(*id_to_index.get(&root_id).ok_or("missing pattern node")?),
        None => None,
    };

    Ok((compiled, root, schema, needs_slow_fallback))
}

fn update_system_var(variables: &mut BTreeMap<String, Value>, key: &str, value: Value) {
    if let Some(slot) = variables.get_mut(key) {
        *slot = value;
    } else {
        variables.insert(key.to_string(), value);
    }
}

pub(crate) fn cook(context: &CookContext) -> CookResult {
    let Some(Value::PatternGraph(graph)) = context.params.get("pattern") else {
        return Err("missing pattern graph".into());
    };

    graph.validate()?;
    let (nodes, root, var_schema, needs_slow_fallback) = compile_pattern_graph(graph, context)?;

    let mut pending = Vec::with_capacity(64);
    if let Some(root) = root {
        pending.push(CompiledVisit { node: root, center: [0.5; 2], extent: 1.0, depth: 0, number: 0, seed: context.seed });
    }

    let mut stamps = Vec::new();
    let mut children_scratch = Vec::with_capacity(8);
    let mut fx_regs = FxRegs::default();
    let mut fx_vars = var_schema.new_store();
    let images = context.inputs.get(1..).unwrap_or(&[]);
    let size = [context.w as f32, context.h as f32];

    // Only maintain the mutable variable map when at least one dynamic parameter uses the slow interpreter
    let mut variables = if needs_slow_fallback {
        let mut vars = context.vars.clone();
        vars.insert("$randomseed".into(), Value::Int(0));
        vars.insert("$depth".into(), Value::Float(0.0));
        vars.insert("$depthpow2".into(), Value::Float(1.0));
        vars.insert("$number".into(), Value::Float(0.0));
        vars
    } else {
        BTreeMap::new()
    };
    let mut state = EvaluationState::default();
    let mut max_depth = 0;
    let mut remaining = MAX_VISITS;

    while let Some(mut visit) = pending.pop() {
        remaining = remaining.checked_sub(1).ok_or("pattern graph exceeded its execution budget")?;
        let node = nodes.get(visit.node).ok_or("missing pattern node")?;
        max_depth = max_depth.max(visit.depth);

        if node.kind == PatternNodeKind::Dot {
            if let Some(child) = node.children[0] {
                visit.node = child;
                pending.push(visit);
            }
            continue;
        }

        if let Some(seed) = node.pre_random_override {
            visit.seed = seed;
        }

        let mut q = node.quadrant;
        let mut switch_val = node.switch_static;
        let mut number_added = node.number_added_static;

        if !node.dynamic.is_empty() {
            let mut random = visit.seed;
            let fx_inputs =
                FxVisitInputs { pos: visit.center, depth: visit.depth as f32, extent: visit.extent, number: visit.number as f32, seed: visit.seed as i32 };
            let mut vars_synced = false;

            for dyn_param in &node.dynamic {
                match &dyn_param.func {
                    CompiledParamFn::Fast(fast) => {
                        let out = fast.eval(&mut fx_regs, &mut fx_vars, &fx_inputs, &mut random, images, context.tile)?;
                        apply_fast_param(dyn_param.slot, out, &mut q, &mut switch_val, &mut number_added)?;
                    }
                    CompiledParamFn::Slow(program) => {
                        if !vars_synced {
                            update_system_var(&mut variables, "$randomseed", Value::Int(visit.seed as i32));
                            update_system_var(&mut variables, "$depth", Value::Float(visit.depth as f32));
                            update_system_var(&mut variables, "$depthpow2", Value::Float(visit.extent));
                            update_system_var(&mut variables, "$number", Value::Float(visit.number as f32));
                            vars_synced = true;
                        }
                        let evaluation = EvaluationContext {
                            scope: crate::function::FunctionScope::FxMap,
                            position: visit.center,
                            size,
                            images,
                            tile: context.tile,
                            variables: &variables,
                        };
                        let value = program.evaluate_with_random(&evaluation, &mut state, &mut random)?;
                        if !state.assignments().is_empty() {
                            for (k, v) in state.assignments() {
                                variables.insert(k.clone(), v.clone());
                            }
                        }
                        apply_value_param(dyn_param.slot, &value, &mut q, &mut switch_val, &mut number_added);
                    }
                }
            }
        }

        if !q.randominherited {
            visit.seed = q.randomseed as u32;
        }

        children_scratch.clear();
        match node.kind {
            PatternNodeKind::Dot => {}
            PatternNodeKind::Quadrant => {
                visit.center = [visit.center[0] + q.branchoffset[0] * visit.extent, visit.center[1] + q.branchoffset[1] * visit.extent];
                let pattern = if q.patterntype == 1 { q.imageindex.max(1).saturating_add(255) } else { q.patterntype };
                if pattern != 0 {
                    if !q.opacity_valid {
                        return Err("pattern color must be a float or float4".into());
                    }
                    stamps.push(PatternStamp {
                        center: [visit.center[0] + q.frameoffset[0] * visit.extent, visit.center[1] + q.frameoffset[1] * visit.extent],
                        size: [q.patternsize[0] * visit.extent, q.patternsize[1] * visit.extent],
                        rotation: q.patternrotation,
                        color: q.color,
                        scalar_color: q.scalar_color,
                        depth: visit.depth,
                        pattern,
                        variation: q.patternsuppl,
                        blend: q.blendingmode,
                        filtering: q.imagefiltering,
                        premultiplied: q.imagepremul,
                    });
                }

                let mut child_random = visit.seed;
                let next_depth = visit.depth.saturating_add(1);
                let next_extent = visit.extent * 0.5;
                let quarter = 0.25 * visit.extent;
                for slot in 0..node.child_count {
                    let child_seed = crate::random::quadrant_child(&mut child_random, next_depth);
                    if let Some(child) = node.children[slot] {
                        children_scratch.push(CompiledVisit {
                            node: child,
                            center: [
                                visit.center[0] + if slot & 1 == 0 { -quarter } else { quarter },
                                visit.center[1] + if slot & 2 == 0 { -quarter } else { quarter },
                            ],
                            extent: next_extent,
                            depth: next_depth,
                            number: visit.number,
                            seed: child_seed,
                        });
                    }
                }
            }
            PatternNodeKind::Switch => {
                let slot = usize::from(switch_val);
                if slot < node.child_count
                    && let Some(child) = node.children[slot]
                {
                    visit.node = child;
                    children_scratch.push(visit);
                }
            }
            PatternNodeKind::Iterate => {
                let count = usize::try_from(number_added).map_err(|_| "iteration count cannot be negative")?;
                if count > remaining || count > 10_000 {
                    return Err("pattern iteration count exceeds the execution budget".into());
                }

                if node.child_count > 0
                    && let Some(child) = node.children[0]
                {
                    children_scratch.push(CompiledVisit {
                        node: child,
                        center: visit.center,
                        extent: visit.extent,
                        depth: visit.depth,
                        number: 0,
                        seed: crate::random::hash(visit.seed.wrapping_add(1)),
                    });
                }
                if node.child_count > 1
                    && let Some(child) = node.children[1]
                {
                    for number in 0..count {
                        children_scratch.push(CompiledVisit {
                            node: child,
                            center: visit.center,
                            extent: visit.extent,
                            depth: visit.depth,
                            number: number as u32,
                            seed: crate::random::hash(visit.seed.wrapping_add(number as u32 + 2)),
                        });
                    }
                }
            }
        }

        if children_scratch.len().saturating_add(pending.len()) > remaining {
            return Err("pattern graph exceeded its execution budget".into());
        }

        pending.extend(children_scratch.iter().rev().copied());
    }

    Ok(vec![raster::render(context, &stamps, max_depth)?])
}

fn apply_fast_param(slot: ParamSlot, out: FxOutput, q: &mut QuadrantStatic, switch_val: &mut bool, number_added: &mut i32) -> Result<(), String> {
    match slot {
        ParamSlot::PatternType => q.patterntype = out.as_i32(),
        ParamSlot::Opacity => match out.as_color() {
            Ok((color, scalar)) => {
                q.color = color;
                q.scalar_color = scalar;
                q.opacity_valid = true;
            }
            Err(_) => q.opacity_valid = false,
        },
        ParamSlot::BranchOffset => q.branchoffset = out.as_vec2(),
        ParamSlot::FrameOffset => q.frameoffset = out.as_vec2(),
        ParamSlot::PatternSize => q.patternsize = out.as_vec2(),
        ParamSlot::PatternRotation => q.patternrotation = out.as_f32(),
        ParamSlot::PatternSuppl => q.patternsuppl = out.as_f32(),
        ParamSlot::BlendingMode => q.blendingmode = out.as_i32(),
        ParamSlot::ImageFiltering => q.imagefiltering = out.as_i32(),
        ParamSlot::ImagePremul => q.imagepremul = out.as_bool(),
        ParamSlot::ImageIndex => q.imageindex = out.as_i32(),
        ParamSlot::RandomSeed => q.randomseed = out.as_i32(),
        ParamSlot::RandomInherited => q.randominherited = out.as_bool(),
        ParamSlot::Switch => *switch_val = out.as_bool(),
        ParamSlot::NumberAdded => *number_added = out.as_i32(),
    }
    Ok(())
}

fn apply_value_param(slot: ParamSlot, value: &Value, q: &mut QuadrantStatic, switch_val: &mut bool, number_added: &mut i32) {
    match slot {
        ParamSlot::PatternType => q.patterntype = value.as_i32().unwrap_or(0),
        ParamSlot::Opacity => match value {
            Value::Float(v) => {
                q.color = [*v, *v, *v, 1.0];
                q.scalar_color = true;
                q.opacity_valid = true;
            }
            other => match other.as_vec4() {
                Some(c) => {
                    q.color = c;
                    q.scalar_color = false;
                    q.opacity_valid = true;
                }
                None => q.opacity_valid = false,
            },
        },
        ParamSlot::BranchOffset => q.branchoffset = value.as_vec4().map(|v| [v[0], v[1]]).unwrap_or([0.0; 2]),
        ParamSlot::FrameOffset => q.frameoffset = value.as_vec4().map(|v| [v[0], v[1]]).unwrap_or([0.0; 2]),
        ParamSlot::PatternSize => q.patternsize = value.as_vec4().map(|v| [v[0], v[1]]).unwrap_or([0.0; 2]),
        ParamSlot::PatternRotation => q.patternrotation = value.as_f32().unwrap_or(0.0),
        ParamSlot::PatternSuppl => q.patternsuppl = value.as_f32().unwrap_or(0.0),
        ParamSlot::BlendingMode => q.blendingmode = value.as_i32().unwrap_or(0),
        ParamSlot::ImageFiltering => q.imagefiltering = value.as_i32().unwrap_or(0),
        ParamSlot::ImagePremul => q.imagepremul = value.as_bool().unwrap_or(false),
        ParamSlot::ImageIndex => q.imageindex = value.as_i32().unwrap_or(0),
        ParamSlot::RandomSeed => q.randomseed = value.as_i32().unwrap_or(0),
        ParamSlot::RandomInherited => q.randominherited = value.as_bool().unwrap_or(false),
        ParamSlot::Switch => *switch_val = value.as_bool().unwrap_or(false),
        ParamSlot::NumberAdded => *number_added = value.as_i32().unwrap_or(0),
    }
}
