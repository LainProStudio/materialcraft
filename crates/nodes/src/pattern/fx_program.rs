//! Compiled zero-allocation scalar bytecode evaluator for FX-Map parameter functions.
//!
//! Acyclic FX-Map parameter functions (including `instance` function calls, `set`/`get` user
//! variables, `sequence`, and short-circuiting `ifelse` branches) are compiled once per cook into
//! flat scalar register operations with a reusable register file and flat variable store.
//! Only graphs with `while` loops, `isdefined`, or non-trivially shared impure conditional nodes
//! fall back to the general `Program` interpreter.

use std::collections::BTreeMap;

use crate::function::{
    EvaluationContext, FunctionScope, Program, SystemVariable,
    value_type::{canonical, integers, type_code},
};
use materialcraft_core::{Value, ValueType};

const MAX_CALL_DEPTH: usize = 16;
const MAX_COMPILED_OPS: usize = 32_768;

#[derive(Clone, Copy, PartialEq, Eq)]
enum RegKind {
    Float,
    Int,
    Bool,
}

#[derive(Clone, Copy)]
struct RegRef {
    kind: RegKind,
    dims: u8,
    slots: [u16; 4],
}

#[derive(Clone, Copy)]
enum FxOp {
    // Register & variable moves
    CopyF { dst: u16, src: u16 },
    CopyI { dst: u16, src: u16 },
    CopyB { dst: u16, src: u16 },
    LoadVarF { dst: u16, var: u16 },
    LoadVarI { dst: u16, var: u16 },
    LoadVarB { dst: u16, var: u16 },
    StoreVarF { var: u16, src: u16 },
    StoreVarI { var: u16, src: u16 },
    StoreVarB { var: u16, src: u16 },
    // Forward-only control flow for lazy `ifelse`
    JumpIfFalse { cond: u16, target: u32 },
    Jump { target: u32 },
    // Float unary
    NegF { dst: u16, a: u16 },
    AbsF { dst: u16, a: u16 },
    FloorF { dst: u16, a: u16 },
    CeilF { dst: u16, a: u16 },
    SqrtF { dst: u16, a: u16 },
    ExpF { dst: u16, a: u16 },
    LogF { dst: u16, a: u16 },
    Log2F { dst: u16, a: u16 },
    Pow2F { dst: u16, a: u16 },
    SinF { dst: u16, a: u16 },
    CosF { dst: u16, a: u16 },
    TanF { dst: u16, a: u16 },
    RandF { dst: u16, a: u16 },
    // Float binary & ternary
    AddF { dst: u16, a: u16, b: u16 },
    SubF { dst: u16, a: u16, b: u16 },
    MulF { dst: u16, a: u16, b: u16 },
    DivF { dst: u16, a: u16, b: u16 },
    ModF { dst: u16, a: u16, b: u16 },
    MinF { dst: u16, a: u16, b: u16 },
    MaxF { dst: u16, a: u16, b: u16 },
    PowF { dst: u16, a: u16, b: u16 },
    Atan2F { dst: u16, y: u16, x: u16 },
    CartesianF { dst_x: u16, dst_y: u16, r: u16, ang: u16 },
    LerpF { dst: u16, a: u16, b: u16, t: u16 },
    // Integer ops
    AddI { dst: u16, a: u16, b: u16 },
    SubI { dst: u16, a: u16, b: u16 },
    MulI { dst: u16, a: u16, b: u16 },
    DivI { dst: u16, a: u16, b: u16 },
    ModI { dst: u16, a: u16, b: u16 },
    NegI { dst: u16, a: u16 },
    AbsI { dst: u16, a: u16 },
    MinI { dst: u16, a: u16, b: u16 },
    MaxI { dst: u16, a: u16, b: u16 },
    IntToFloat { dst: u16, a: u16 },
    FloatToInt { dst: u16, a: u16 },
    IntToBool { dst: u16, a: u16 },
    // Comparisons & logic
    EqF { dst: u16, a: u16, b: u16 },
    EqI { dst: u16, a: u16, b: u16 },
    EqB { dst: u16, a: u16, b: u16 },
    GtF { dst: u16, a: u16, b: u16 },
    GtEqF { dst: u16, a: u16, b: u16 },
    LrF { dst: u16, a: u16, b: u16 },
    LrEqF { dst: u16, a: u16, b: u16 },
    GtI { dst: u16, a: u16, b: u16 },
    GtEqI { dst: u16, a: u16, b: u16 },
    LrI { dst: u16, a: u16, b: u16 },
    LrEqI { dst: u16, a: u16, b: u16 },
    GtB { dst: u16, a: u16, b: u16 },
    GtEqB { dst: u16, a: u16, b: u16 },
    LrB { dst: u16, a: u16, b: u16 },
    LrEqB { dst: u16, a: u16, b: u16 },
    And { dst: u16, a: u16, b: u16 },
    Or { dst: u16, a: u16, b: u16 },
    Not { dst: u16, a: u16 },
    // Texture sampling
    SampleLum { dst: u16, u: u16, v: u16, index: u16, bilinear: bool },
    SampleCol { dst: [u16; 4], u: u16, v: u16, index: u16, bilinear: bool },
}

#[derive(Clone, Copy)]
struct FxVarInfo {
    kind: RegKind,
    dims: u8,
    type_code: u32,
    slots: [u16; 4],
}

/// Shared flat storage layout for user variables written by `set` across an FX-Map cook.
pub(super) struct FxVarSchema {
    by_name: BTreeMap<String, FxVarInfo>,
    f_init: Vec<f32>,
    i_init: Vec<i32>,
    b_init: Vec<bool>,
    valid: bool,
}

impl FxVarSchema {
    pub(super) fn empty() -> Self {
        Self { by_name: BTreeMap::new(), f_init: Vec::new(), i_init: Vec::new(), b_init: Vec::new(), valid: true }
    }

    pub(super) fn collect<'a>(programs: impl IntoIterator<Item = &'a Program>, vars: &BTreeMap<String, Value>) -> Self {
        let mut schema = Self::empty();
        for program in programs {
            if !schema.collect_program(program, vars, 0) {
                schema.valid = false;
                break;
            }
        }
        schema
    }

    fn collect_program(&mut self, program: &Program, vars: &BTreeMap<String, Value>, depth: usize) -> bool {
        if depth > MAX_CALL_DEPTH {
            return false;
        }
        for inst in &program.instructions {
            if let Some(call) = &inst.call {
                if call.exports && !self.collect_program(&call.program, vars, depth + 1) {
                    return false;
                }
                continue;
            }
            if inst.definition.id == "set" {
                let Some(Value::Text(name)) = &inst.value else { return false };
                if name.is_empty() || name.starts_with('$') {
                    return false;
                }
                let Some((kind, dims)) = reg_from_type_code(inst.type_code) else { return false };
                if let Some(existing) = self.by_name.get(name) {
                    if existing.type_code != inst.type_code {
                        return false;
                    }
                    continue;
                }
                let init_val = match vars.get(name.as_str()).cloned() {
                    Some(v) => {
                        if type_code(&v) != inst.type_code {
                            return false;
                        }
                        let Ok(c) = canonical(v) else { return false };
                        c
                    }
                    None => {
                        let Some(vt) = ValueType::from_code(inst.type_code).filter(|k| *k != ValueType::String) else {
                            return false;
                        };
                        vt.default_value()
                    }
                };
                let Some(slots) = self.alloc_var_slots(kind, dims, &init_val) else { return false };
                self.by_name.insert(name.clone(), FxVarInfo { kind, dims, type_code: inst.type_code, slots });
            }
        }
        true
    }

    fn alloc_var_slots(&mut self, kind: RegKind, dims: u8, init: &Value) -> Option<[u16; 4]> {
        let mut slots = [0u16; 4];
        match kind {
            RegKind::Bool => {
                let id = u16::try_from(self.b_init.len()).ok()?;
                self.b_init.push(init.as_bool().unwrap_or(false));
                slots[0] = id;
            }
            RegKind::Int => {
                let vals = integers(init).ok()?;
                for k in 0..usize::from(dims) {
                    let id = u16::try_from(self.i_init.len()).ok()?;
                    self.i_init.push(*vals.get(k)?);
                    *slots.get_mut(k)? = id;
                }
            }
            RegKind::Float => {
                let vals = match init {
                    Value::Float(v) => [*v, 0.0, 0.0, 0.0],
                    Value::Float2(v) => [v[0], v[1], 0.0, 0.0],
                    Value::Float3(v) => [v[0], v[1], v[2], 0.0],
                    Value::Float4(v) | Value::Color(v) => *v,
                    _ => return None,
                };
                for k in 0..usize::from(dims) {
                    let id = u16::try_from(self.f_init.len()).ok()?;
                    self.f_init.push(*vals.get(k)?);
                    *slots.get_mut(k)? = id;
                }
            }
        }
        Some(slots)
    }

    pub(super) fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }

    pub(super) fn is_valid(&self) -> bool {
        self.valid
    }

    pub(super) fn new_store(&self) -> FxVarStore {
        FxVarStore { f: self.f_init.clone(), i: self.i_init.clone(), b: self.b_init.clone() }
    }
}

#[derive(Default)]
pub(super) struct FxVarStore {
    f: Vec<f32>,
    i: Vec<i32>,
    b: Vec<bool>,
}

#[derive(Default)]
pub(super) struct FxRegs {
    f: Vec<f32>,
    i: Vec<i32>,
    b: Vec<bool>,
}

pub(super) struct FxVisitInputs {
    pub pos: [f32; 2],
    pub depth: f32,
    pub extent: f32,
    pub number: f32,
    pub seed: i32,
}

#[derive(Clone, Copy)]
pub(super) enum FxOutput {
    Bool(bool),
    Int(i32),
    Float(f32),
    Float2([f32; 2]),
    Float3([f32; 3]),
    Float4([f32; 4]),
}

impl FxOutput {
    pub(super) fn as_f32(self) -> f32 {
        match self {
            Self::Float(v) => v,
            Self::Int(v) => v as f32,
            _ => 0.0,
        }
    }

    pub(super) fn as_i32(self) -> i32 {
        match self {
            Self::Int(v) => v,
            Self::Float(v) => v as i32,
            _ => 0,
        }
    }

    pub(super) fn as_bool(self) -> bool {
        match self {
            Self::Bool(v) => v,
            _ => false,
        }
    }

    pub(super) fn as_vec2(self) -> [f32; 2] {
        match self {
            Self::Float2(v) => v,
            Self::Float3(v) => [v[0], v[1]],
            Self::Float4(v) => [v[0], v[1]],
            Self::Float(v) => [v, v],
            _ => [0.0; 2],
        }
    }

    pub(super) fn as_color(self) -> Result<([f32; 4], bool), String> {
        match self {
            Self::Float(v) => Ok(([v, v, v, 1.0], true)),
            Self::Float2(v) => Ok(([v[0], v[1], 0.0, 1.0], false)),
            Self::Float3(v) => Ok(([v[0], v[1], v[2], 1.0], false)),
            Self::Float4(v) => Ok((v, false)),
            _ => Err("pattern color must be a float or float4".into()),
        }
    }
}

pub(super) struct FxProgram {
    ops: Vec<FxOp>,
    f_init: Vec<f32>,
    i_init: Vec<i32>,
    b_init: Vec<bool>,
    pos_x: u16,
    pos_y: u16,
    depth: u16,
    extent: u16,
    number: u16,
    seed: u16,
    output: RegRef,
}

struct Builder {
    ops: Vec<FxOp>,
    f_init: Vec<f32>,
    f_const: Vec<Option<f32>>,
    i_init: Vec<i32>,
    i_const: Vec<Option<i32>>,
    b_init: Vec<bool>,
    b_const: Vec<Option<bool>>,
}

impl Builder {
    fn new() -> Self {
        Self { ops: Vec::new(), f_init: Vec::new(), f_const: Vec::new(), i_init: Vec::new(), i_const: Vec::new(), b_init: Vec::new(), b_const: Vec::new() }
    }

    fn push_op(&mut self, op: FxOp) -> Option<usize> {
        if self.ops.len() >= MAX_COMPILED_OPS {
            return None;
        }
        let idx = self.ops.len();
        self.ops.push(op);
        Some(idx)
    }

    fn alloc_f_dyn(&mut self) -> Option<u16> {
        let id = u16::try_from(self.f_init.len()).ok()?;
        self.f_init.push(0.0);
        self.f_const.push(None);
        Some(id)
    }

    fn alloc_f_const(&mut self, v: f32) -> Option<u16> {
        let id = u16::try_from(self.f_init.len()).ok()?;
        self.f_init.push(v);
        self.f_const.push(Some(v));
        Some(id)
    }

    fn alloc_i_dyn(&mut self) -> Option<u16> {
        let id = u16::try_from(self.i_init.len()).ok()?;
        self.i_init.push(0);
        self.i_const.push(None);
        Some(id)
    }

    fn alloc_i_const(&mut self, v: i32) -> Option<u16> {
        let id = u16::try_from(self.i_init.len()).ok()?;
        self.i_init.push(v);
        self.i_const.push(Some(v));
        Some(id)
    }

    fn alloc_b_dyn(&mut self) -> Option<u16> {
        let id = u16::try_from(self.b_init.len()).ok()?;
        self.b_init.push(false);
        self.b_const.push(None);
        Some(id)
    }

    fn alloc_b_const(&mut self, v: bool) -> Option<u16> {
        let id = u16::try_from(self.b_init.len()).ok()?;
        self.b_init.push(v);
        self.b_const.push(Some(v));
        Some(id)
    }

    fn alloc_dyn_ref(&mut self, kind: RegKind, dims: u8) -> Option<RegRef> {
        let mut slots = [0u16; 4];
        for k in 0..usize::from(dims) {
            *slots.get_mut(k)? = match kind {
                RegKind::Float => self.alloc_f_dyn()?,
                RegKind::Int => self.alloc_i_dyn()?,
                RegKind::Bool => self.alloc_b_dyn()?,
            };
        }
        Some(RegRef { kind, dims, slots })
    }

    fn emit_copy(&mut self, dst: RegRef, src: RegRef) -> Option<()> {
        if dst.kind != src.kind || dst.dims != src.dims {
            return None;
        }
        for k in 0..usize::from(dst.dims) {
            let d = *dst.slots.get(k)?;
            let s = *src.slots.get(k)?;
            if d == s {
                continue;
            }
            let op = match dst.kind {
                RegKind::Float => FxOp::CopyF { dst: d, src: s },
                RegKind::Int => FxOp::CopyI { dst: d, src: s },
                RegKind::Bool => FxOp::CopyB { dst: d, src: s },
            };
            self.push_op(op)?;
        }
        Some(())
    }

    fn emit_load_var(&mut self, info: FxVarInfo) -> Option<RegRef> {
        let dst = self.alloc_dyn_ref(info.kind, info.dims)?;
        for k in 0..usize::from(info.dims) {
            let d = *dst.slots.get(k)?;
            let v = *info.slots.get(k)?;
            let op = match info.kind {
                RegKind::Float => FxOp::LoadVarF { dst: d, var: v },
                RegKind::Int => FxOp::LoadVarI { dst: d, var: v },
                RegKind::Bool => FxOp::LoadVarB { dst: d, var: v },
            };
            self.push_op(op)?;
        }
        Some(dst)
    }

    fn emit_store_var(&mut self, info: FxVarInfo, src: RegRef) -> Option<()> {
        if info.kind != src.kind || info.dims != src.dims {
            return None;
        }
        for k in 0..usize::from(info.dims) {
            let v = *info.slots.get(k)?;
            let s = *src.slots.get(k)?;
            let op = match info.kind {
                RegKind::Float => FxOp::StoreVarF { var: v, src: s },
                RegKind::Int => FxOp::StoreVarI { var: v, src: s },
                RegKind::Bool => FxOp::StoreVarB { var: v, src: s },
            };
            self.push_op(op)?;
        }
        Some(())
    }

    fn f_c(&self, slot: u16) -> Option<f32> {
        self.f_const.get(usize::from(slot)).copied().flatten()
    }

    fn i_c(&self, slot: u16) -> Option<i32> {
        self.i_const.get(usize::from(slot)).copied().flatten()
    }

    fn b_c(&self, slot: u16) -> Option<bool> {
        self.b_const.get(usize::from(slot)).copied().flatten()
    }

    fn const_value(&mut self, value: &Value) -> Option<RegRef> {
        match value {
            Value::Bool(v) => Some(RegRef { kind: RegKind::Bool, dims: 1, slots: [self.alloc_b_const(*v)?, 0, 0, 0] }),
            Value::Int(v) => Some(RegRef { kind: RegKind::Int, dims: 1, slots: [self.alloc_i_const(*v)?, 0, 0, 0] }),
            Value::Int2(v) => Some(RegRef { kind: RegKind::Int, dims: 2, slots: [self.alloc_i_const(v[0])?, self.alloc_i_const(v[1])?, 0, 0] }),
            Value::Int3(v) => {
                Some(RegRef { kind: RegKind::Int, dims: 3, slots: [self.alloc_i_const(v[0])?, self.alloc_i_const(v[1])?, self.alloc_i_const(v[2])?, 0] })
            }
            Value::Int4(v) => Some(RegRef {
                kind: RegKind::Int,
                dims: 4,
                slots: [self.alloc_i_const(v[0])?, self.alloc_i_const(v[1])?, self.alloc_i_const(v[2])?, self.alloc_i_const(v[3])?],
            }),
            Value::Enum(v) => Some(RegRef { kind: RegKind::Int, dims: 1, slots: [self.alloc_i_const(i32::try_from(*v).ok()?)?, 0, 0, 0] }),
            Value::Float(v) => Some(RegRef { kind: RegKind::Float, dims: 1, slots: [self.alloc_f_const(*v)?, 0, 0, 0] }),
            Value::Float2(v) => Some(RegRef { kind: RegKind::Float, dims: 2, slots: [self.alloc_f_const(v[0])?, self.alloc_f_const(v[1])?, 0, 0] }),
            Value::Float3(v) => {
                Some(RegRef { kind: RegKind::Float, dims: 3, slots: [self.alloc_f_const(v[0])?, self.alloc_f_const(v[1])?, self.alloc_f_const(v[2])?, 0] })
            }
            Value::Float4(v) | Value::Color(v) => Some(RegRef {
                kind: RegKind::Float,
                dims: 4,
                slots: [self.alloc_f_const(v[0])?, self.alloc_f_const(v[1])?, self.alloc_f_const(v[2])?, self.alloc_f_const(v[3])?],
            }),
            _ => None,
        }
    }

    fn emit_unary_f(&mut self, op: &str, a: u16) -> Option<u16> {
        if let Some(va) = self.f_c(a) {
            let r = match op {
                "neg" => -va,
                "abs" => va.abs(),
                "floor" => va.floor(),
                "ceil" => va.ceil(),
                "sqrt" => va.sqrt(),
                "exp" => va.exp(),
                "log" => va.ln(),
                "log2" => va.log2(),
                "pow2" => va.exp2(),
                "sin" => va.sin(),
                "cos" => va.cos(),
                _ => va.tan(),
            };
            return self.alloc_f_const(r);
        }

        let dst = self.alloc_f_dyn()?;
        let fx_op = match op {
            "neg" => FxOp::NegF { dst, a },
            "abs" => FxOp::AbsF { dst, a },
            "floor" => FxOp::FloorF { dst, a },
            "ceil" => FxOp::CeilF { dst, a },
            "sqrt" => FxOp::SqrtF { dst, a },
            "exp" => FxOp::ExpF { dst, a },
            "log" => FxOp::LogF { dst, a },
            "log2" => FxOp::Log2F { dst, a },
            "pow2" => FxOp::Pow2F { dst, a },
            "sin" => FxOp::SinF { dst, a },
            "cos" => FxOp::CosF { dst, a },
            _ => FxOp::TanF { dst, a },
        };
        self.push_op(fx_op)?;
        Some(dst)
    }

    fn emit_bin_f(&mut self, op: &str, a: u16, b: u16) -> Option<u16> {
        if let (Some(va), Some(vb)) = (self.f_c(a), self.f_c(b)) {
            let r = match op {
                "add" => va + vb,
                "sub" => va - vb,
                "mul" => va * vb,
                "div" => va / vb,
                "mod" => va - vb * (va / vb).floor(),
                "min" => va.min(vb),
                "max" => va.max(vb),
                _ => va.powf(vb),
            };
            return self.alloc_f_const(r);
        }

        let dst = self.alloc_f_dyn()?;
        let fx_op = match op {
            "add" => FxOp::AddF { dst, a, b },
            "sub" => FxOp::SubF { dst, a, b },
            "mul" => FxOp::MulF { dst, a, b },
            "div" => FxOp::DivF { dst, a, b },
            "mod" => FxOp::ModF { dst, a, b },
            "min" => FxOp::MinF { dst, a, b },
            "max" => FxOp::MaxF { dst, a, b },
            _ => FxOp::PowF { dst, a, b },
        };
        self.push_op(fx_op)?;
        Some(dst)
    }

    fn emit_and(&mut self, a: u16, b: u16) -> Option<u16> {
        if let (Some(va), Some(vb)) = (self.b_c(a), self.b_c(b)) {
            return self.alloc_b_const(va && vb);
        }
        let dst = self.alloc_b_dyn()?;
        self.push_op(FxOp::And { dst, a, b })?;
        Some(dst)
    }

    fn emit_or(&mut self, a: u16, b: u16) -> Option<u16> {
        if let (Some(va), Some(vb)) = (self.b_c(a), self.b_c(b)) {
            return self.alloc_b_const(va || vb);
        }
        let dst = self.alloc_b_dyn()?;
        self.push_op(FxOp::Or { dst, a, b })?;
        Some(dst)
    }

    fn emit_not(&mut self, a: u16) -> Option<u16> {
        if let Some(va) = self.b_c(a) {
            return self.alloc_b_const(!va);
        }
        let dst = self.alloc_b_dyn()?;
        self.push_op(FxOp::Not { dst, a })?;
        Some(dst)
    }
}

#[derive(Clone)]
enum LocalBinding {
    /// Immutable parameter binding from `call.inputs`
    Param(RegRef),
    /// Lazy constant default from `call.defaults`
    Default(Value),
    /// Mutable register slot for a local `set` inside a non-exporting callee
    Mutable(RegRef),
}

struct Scope<'a> {
    parent: Option<&'a Scope<'a>>,
    locals: BTreeMap<String, LocalBinding>,
    /// True iff `set` in this scope exports into the caller/shared FX-Map variable store
    exporting: bool,
}

impl Scope<'_> {
    fn lookup(&self, name: &str) -> Option<&LocalBinding> {
        if let Some(binding) = self.locals.get(name) {
            return Some(binding);
        }
        self.parent.and_then(|p| p.lookup(name))
    }

    /// Returns true if `name` is shadowed as a `Param` or `Default` along an exporting chain
    /// (between this scope and the target store) which would diverge from direct store writes.
    fn shadows_export_target(&self, name: &str) -> bool {
        if !self.exporting {
            return false;
        }
        if matches!(self.locals.get(name), Some(LocalBinding::Param(_) | LocalBinding::Default(_))) {
            return true;
        }
        self.parent.is_some_and(|p| p.shadows_export_target(name))
    }

    /// Find the enclosing non-exporting scope's mutable slot for `name`, if any.
    fn lookup_local_mutable_target(&self, name: &str) -> Option<RegRef> {
        if !self.exporting {
            return match self.locals.get(name) {
                Some(LocalBinding::Mutable(r)) => Some(*r),
                _ => None,
            };
        }
        self.parent.and_then(|p| p.lookup_local_mutable_target(name))
    }

    fn writes_to_shared_store(&self) -> bool {
        if !self.exporting {
            return false;
        }
        self.parent.is_none_or(Scope::writes_to_shared_store)
    }
}

struct CompileEnv<'a> {
    b: Builder,
    schema: &'a FxVarSchema,
    vars: &'a BTreeMap<String, Value>,
    dummy_ctx: EvaluationContext<'a>,
    pos_ref: RegRef,
    depth_ref: RegRef,
    extent_ref: RegRef,
    number_ref: RegRef,
    seed_ref: RegRef,
    zero_f: u16,
    zero_i: u16,
}

impl<'a> CompileEnv<'a> {
    fn resolve_variable(&mut self, scope: &Scope<'_>, op: &str, name: &str, expected_type: u32) -> Option<(RegRef, bool)> {
        match name {
            "$pos" => {
                if expected_type != 512 {
                    return None;
                }
                return Some((self.pos_ref, false));
            }
            "$depth" => {
                if expected_type != 256 {
                    return None;
                }
                return Some((self.depth_ref, false));
            }
            "$depthpow2" => {
                if expected_type != 256 {
                    return None;
                }
                return Some((self.extent_ref, false));
            }
            "$number" => {
                if expected_type != 256 {
                    return None;
                }
                return Some((self.number_ref, false));
            }
            "$randomseed" => {
                if expected_type != 16 {
                    return None;
                }
                return Some((self.seed_ref, false));
            }
            _ if SystemVariable::named(name).is_some() => {
                let resolved = self.dummy_ctx.variable(name, &BTreeMap::new()).ok()??;
                if type_code(&resolved) != expected_type {
                    return None;
                }
                let canonical = canonical(resolved).ok()?;
                return Some((self.b.const_value(&canonical)?, false));
            }
            _ => {}
        }

        if let Some(binding) = scope.lookup(name) {
            return match binding {
                LocalBinding::Param(r) => {
                    let r = *r;
                    Some((self.coerce_get_reg(op, name, r, expected_type)?, false))
                }
                LocalBinding::Default(val) => Some((self.coerce_get_value(op, name, val.clone(), expected_type)?, false)),
                LocalBinding::Mutable(r) => {
                    let r = *r;
                    let fresh = self.b.alloc_dyn_ref(r.kind, r.dims)?;
                    self.b.emit_copy(fresh, r)?;
                    Some((self.coerce_get_reg(op, name, fresh, expected_type)?, true))
                }
            };
        }

        if let Some(&info) = self.schema.by_name.get(name) {
            let loaded = self.b.emit_load_var(info)?;
            return Some((self.coerce_get_reg(op, name, loaded, expected_type)?, true));
        }

        let val = match self.vars.get(name).cloned() {
            Some(v) => v,
            None => {
                if name.starts_with('$') {
                    return None;
                }
                ValueType::from_code(expected_type).filter(|k| *k != ValueType::String)?.default_value()
            }
        };
        Some((self.coerce_get_value(op, name, val, expected_type)?, false))
    }

    fn coerce_get_reg(&mut self, op: &str, name: &str, r: RegRef, expected_type: u32) -> Option<RegRef> {
        if op == "get_bool" && !name.starts_with('$') && r.kind == RegKind::Int && r.dims == 1 {
            let src = r.slots[0];
            let dst = if let Some(vi) = self.b.i_c(src) {
                self.b.alloc_b_const(vi != 0)?
            } else {
                let d = self.b.alloc_b_dyn()?;
                self.b.push_op(FxOp::IntToBool { dst: d, a: src })?;
                d
            };
            return Some(RegRef { kind: RegKind::Bool, dims: 1, slots: [dst, 0, 0, 0] });
        }
        if reg_type_code(r.kind, r.dims) != expected_type {
            return None;
        }
        Some(r)
    }

    fn coerce_get_value(&mut self, op: &str, name: &str, val: Value, expected_type: u32) -> Option<RegRef> {
        let val =
            if op == "get_bool" && !name.starts_with('$') && matches!(val, Value::Int(_) | Value::Enum(_)) { Value::Bool(val.as_i32()? != 0) } else { val };
        if type_code(&val) != expected_type {
            return None;
        }
        let canonical = canonical(val).ok()?;
        self.b.const_value(&canonical)
    }

    fn compile_program(&mut self, program: &Program, scope: &Scope<'_>, call_depth: usize, in_branch: usize) -> Option<RegRef> {
        let len = program.instructions.len();
        if len == 0 || len > usize::from(u16::MAX) || call_depth > MAX_CALL_DEPTH {
            return None;
        }

        let mut memo: Vec<Option<RegRef>> = vec![None; len];
        let mut branch_impure: Vec<bool> = vec![false; len];
        let output = self.compile_node(program, program.output, scope, &mut memo, &mut branch_impure, call_depth, in_branch)?;

        let expected_type = program.instructions.get(program.output)?.type_code;
        if reg_type_code(output.kind, output.dims) != expected_type {
            return None;
        }
        Some(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn compile_node(
        &mut self,
        program: &Program,
        idx: usize,
        scope: &Scope<'_>,
        memo: &mut [Option<RegRef>],
        branch_impure: &mut [bool],
        call_depth: usize,
        in_branch: usize,
    ) -> Option<RegRef> {
        if let Some(cached) = *memo.get(idx)? {
            return Some(cached);
        }

        let inst = program.instructions.get(idx)?;

        // Handle function instance calls (inlined into the caller bytecode)
        if let Some(call) = &inst.call {
            if *branch_impure.get(idx)? {
                return None;
            }
            let mut args = Vec::with_capacity(inst.inputs.len());
            for &src in &inst.inputs {
                args.push(self.compile_node(program, src, scope, memo, branch_impure, call_depth, in_branch)?);
            }
            if in_branch > 0 && program_has_impure_effects(&call.program, self.schema, scope, call_depth + 1) {
                *branch_impure.get_mut(idx)? = true;
            }
            let res = self.compile_call(call, &args, scope, call_depth + 1, in_branch)?;
            *memo.get_mut(idx)? = Some(res);
            return Some(res);
        }

        let op = inst.definition.id;

        // Control-flow and sequencing ops evaluate inputs lazily in exact interpreter order
        if op == "sequence" {
            let first = *inst.inputs.first()?;
            let second = *inst.inputs.get(1)?;
            let _ = self.compile_node(program, first, scope, memo, branch_impure, call_depth, in_branch)?;
            let res = self.compile_node(program, second, scope, memo, branch_impure, call_depth, in_branch)?;
            *memo.get_mut(idx)? = Some(res);
            return Some(res);
        }

        if op == "ifelse" {
            let cond_idx = *inst.inputs.first()?;
            let then_idx = *inst.inputs.get(1)?;
            let else_idx = *inst.inputs.get(2)?;
            let rc = self.compile_node(program, cond_idx, scope, memo, branch_impure, call_depth, in_branch)?;
            if rc.kind != RegKind::Bool || rc.dims != 1 {
                return None;
            }
            let res = if let Some(cv) = self.b.b_c(rc.slots[0]) {
                self.compile_node(program, if cv { then_idx } else { else_idx }, scope, memo, branch_impure, call_depth, in_branch)?
            } else {
                let jif_pos = self.b.push_op(FxOp::JumpIfFalse { cond: rc.slots[0], target: 0 })?;

                let mut then_memo = memo.to_vec();
                let mut then_impure = branch_impure.to_vec();
                let rt = self.compile_node(program, then_idx, scope, &mut then_memo, &mut then_impure, call_depth, in_branch + 1)?;
                let out_ref = self.b.alloc_dyn_ref(rt.kind, rt.dims)?;
                self.b.emit_copy(out_ref, rt)?;
                let jmp_pos = self.b.push_op(FxOp::Jump { target: 0 })?;

                let else_start = u32::try_from(self.b.ops.len()).ok()?;
                if let Some(FxOp::JumpIfFalse { target, .. }) = self.b.ops.get_mut(jif_pos) {
                    *target = else_start;
                }

                let mut else_memo = memo.to_vec();
                let mut else_impure = branch_impure.to_vec();
                let re = self.compile_node(program, else_idx, scope, &mut else_memo, &mut else_impure, call_depth, in_branch + 1)?;
                if re.kind != rt.kind || re.dims != rt.dims {
                    return None;
                }
                self.b.emit_copy(out_ref, re)?;

                let end_pc = u32::try_from(self.b.ops.len()).ok()?;
                if let Some(FxOp::Jump { target }) = self.b.ops.get_mut(jmp_pos) {
                    *target = end_pc;
                }

                for (dst, (&t, &e)) in branch_impure.iter_mut().zip(then_impure.iter().zip(else_impure.iter())) {
                    *dst |= t | e;
                }
                out_ref
            };
            *memo.get_mut(idx)? = Some(res);
            return Some(res);
        }

        if matches!(op, "while" | "isdefined") {
            return None;
        }

        // Evaluate inputs left-to-right (matching Program::run pushing inputs.rev())
        let mut inputs = [RegRef { kind: RegKind::Float, dims: 0, slots: [0; 4] }; 3];
        if inst.inputs.len() > inputs.len() {
            return None;
        }
        for (slot, &src) in inst.inputs.iter().enumerate() {
            *inputs.get_mut(slot)? = self.compile_node(program, src, scope, memo, branch_impure, call_depth, in_branch)?;
        }

        let in_ref = |slot: usize| -> Option<RegRef> { if slot < inst.inputs.len() { inputs.get(slot).copied() } else { None } };

        // Guard against re-evaluating an impure/trapping node that was already emitted inside a branch
        let mut is_impure = false;

        let out = match op {
            "passthrough" => in_ref(0)?,
            op if op.starts_with("const_") => {
                let val = inst.value.as_ref()?;
                if type_code(val) != inst.type_code {
                    return None;
                }
                let canonical = canonical(val.clone()).ok()?;
                self.b.const_value(&canonical)?
            }
            op if op.starts_with("get_") => {
                let Some(Value::Text(name)) = &inst.value else { return None };
                let (reg, reads_mut) = self.resolve_variable(scope, op, name, inst.type_code)?;
                is_impure = reads_mut;
                reg
            }
            "set" => {
                is_impure = true;
                let Some(Value::Text(name)) = &inst.value else { return None };
                if name.is_empty() || name.starts_with('$') {
                    return None;
                }
                let ra = in_ref(0)?;
                if reg_type_code(ra.kind, ra.dims) != inst.type_code {
                    return None;
                }
                if scope.shadows_export_target(name) {
                    return None;
                }
                if scope.writes_to_shared_store() {
                    let info = *self.schema.by_name.get(name)?;
                    self.b.emit_store_var(info, ra)?;
                } else {
                    let target = scope.lookup_local_mutable_target(name)?;
                    self.b.emit_copy(target, ra)?;
                }
                ra
            }
            "rand" => {
                is_impure = true;
                let ra = in_ref(0)?;
                if ra.kind != RegKind::Float {
                    return None;
                }
                let mut slots = [0u16; 4];
                for k in 0..usize::from(ra.dims) {
                    let dst = self.b.alloc_f_dyn()?;
                    self.b.push_op(FxOp::RandF { dst, a: *ra.slots.get(k)? })?;
                    *slots.get_mut(k)? = dst;
                }
                RegRef { kind: RegKind::Float, dims: ra.dims, slots }
            }
            "neg" | "abs" => {
                let ra = in_ref(0)?;
                match ra.kind {
                    RegKind::Float => {
                        let mut slots = [0u16; 4];
                        for k in 0..usize::from(ra.dims) {
                            *slots.get_mut(k)? = self.b.emit_unary_f(op, *ra.slots.get(k)?)?;
                        }
                        RegRef { kind: RegKind::Float, dims: ra.dims, slots }
                    }
                    RegKind::Int => {
                        is_impure = true;
                        let mut slots = [0u16; 4];
                        for k in 0..usize::from(ra.dims) {
                            let sa = *ra.slots.get(k)?;
                            let dst = if let Some(va) = self.b.i_c(sa) {
                                let r = if op == "neg" { va.checked_neg()? } else { va.checked_abs()? };
                                self.b.alloc_i_const(r)?
                            } else {
                                let dst = self.b.alloc_i_dyn()?;
                                self.b.push_op(if op == "neg" { FxOp::NegI { dst, a: sa } } else { FxOp::AbsI { dst, a: sa } })?;
                                dst
                            };
                            *slots.get_mut(k)? = dst;
                        }
                        RegRef { kind: RegKind::Int, dims: ra.dims, slots }
                    }
                    RegKind::Bool => return None,
                }
            }
            "floor" | "ceil" | "sqrt" | "exp" | "log" | "log2" | "pow2" | "sin" | "cos" | "tan" => {
                let ra = in_ref(0)?;
                if ra.kind != RegKind::Float {
                    return None;
                }
                let mut slots = [0u16; 4];
                for k in 0..usize::from(ra.dims) {
                    *slots.get_mut(k)? = self.b.emit_unary_f(op, *ra.slots.get(k)?)?;
                }
                RegRef { kind: RegKind::Float, dims: ra.dims, slots }
            }
            "add" | "sub" | "mul" | "div" | "mod" | "min" | "max" | "pow" => {
                let ra = in_ref(0)?;
                let rb = in_ref(1)?;
                if ra.kind != rb.kind || ra.dims != rb.dims {
                    return None;
                }
                match ra.kind {
                    RegKind::Float => {
                        let mut slots = [0u16; 4];
                        for k in 0..usize::from(ra.dims) {
                            *slots.get_mut(k)? = self.b.emit_bin_f(op, *ra.slots.get(k)?, *rb.slots.get(k)?)?;
                        }
                        RegRef { kind: RegKind::Float, dims: ra.dims, slots }
                    }
                    RegKind::Int => {
                        if op == "pow" {
                            return None;
                        }
                        if !matches!(op, "min" | "max") {
                            is_impure = true;
                        }
                        let mut slots = [0u16; 4];
                        for k in 0..usize::from(ra.dims) {
                            let sa = *ra.slots.get(k)?;
                            let sb = *rb.slots.get(k)?;
                            let dst = if let (Some(va), Some(vb)) = (self.b.i_c(sa), self.b.i_c(sb)) {
                                let r = match op {
                                    "add" => va.checked_add(vb)?,
                                    "sub" => va.checked_sub(vb)?,
                                    "mul" => va.checked_mul(vb)?,
                                    "div" => va.checked_div(vb)?,
                                    "mod" => va.checked_rem(vb)?,
                                    "min" => va.min(vb),
                                    _ => va.max(vb),
                                };
                                self.b.alloc_i_const(r)?
                            } else {
                                let dst = self.b.alloc_i_dyn()?;
                                let fx_op = match op {
                                    "add" => FxOp::AddI { dst, a: sa, b: sb },
                                    "sub" => FxOp::SubI { dst, a: sa, b: sb },
                                    "mul" => FxOp::MulI { dst, a: sa, b: sb },
                                    "div" => FxOp::DivI { dst, a: sa, b: sb },
                                    "mod" => FxOp::ModI { dst, a: sa, b: sb },
                                    "min" => FxOp::MinI { dst, a: sa, b: sb },
                                    _ => FxOp::MaxI { dst, a: sa, b: sb },
                                };
                                self.b.push_op(fx_op)?;
                                dst
                            };
                            *slots.get_mut(k)? = dst;
                        }
                        RegRef { kind: RegKind::Int, dims: ra.dims, slots }
                    }
                    RegKind::Bool => return None,
                }
            }
            "mulscalar" => {
                let ra = in_ref(0)?;
                let rb = in_ref(1)?;
                if ra.kind != RegKind::Float || rb.kind != RegKind::Float || rb.dims != 1 {
                    return None;
                }
                let s = rb.slots[0];
                let mut slots = [0u16; 4];
                for k in 0..usize::from(ra.dims) {
                    *slots.get_mut(k)? = self.b.emit_bin_f("mul", *ra.slots.get(k)?, s)?;
                }
                RegRef { kind: RegKind::Float, dims: ra.dims, slots }
            }
            "dot" => {
                let ra = in_ref(0)?;
                let rb = in_ref(1)?;
                if ra.kind != RegKind::Float || rb.kind != RegKind::Float || ra.dims != rb.dims || ra.dims == 0 {
                    return None;
                }
                let mut acc = self.b.emit_bin_f("mul", ra.slots[0], rb.slots[0])?;
                for k in 1..usize::from(ra.dims) {
                    let prod = self.b.emit_bin_f("mul", *ra.slots.get(k)?, *rb.slots.get(k)?)?;
                    acc = self.b.emit_bin_f("add", acc, prod)?;
                }
                RegRef { kind: RegKind::Float, dims: 1, slots: [acc, 0, 0, 0] }
            }
            "atan2" => {
                let ra = in_ref(0)?;
                if ra.kind != RegKind::Float || ra.dims != 2 {
                    return None;
                }
                let dst = if let (Some(vx), Some(vy)) = (self.b.f_c(ra.slots[0]), self.b.f_c(ra.slots[1])) {
                    self.b.alloc_f_const(vy.atan2(vx))?
                } else {
                    let dst = self.b.alloc_f_dyn()?;
                    self.b.push_op(FxOp::Atan2F { dst, y: ra.slots[1], x: ra.slots[0] })?;
                    dst
                };
                RegRef { kind: RegKind::Float, dims: 1, slots: [dst, 0, 0, 0] }
            }
            "cartesian" => {
                let ra = in_ref(0)?;
                let rb = in_ref(1)?;
                if ra.kind != RegKind::Float || rb.kind != RegKind::Float || ra.dims != 1 || rb.dims != 1 {
                    return None;
                }
                let (dx, dy) = if let (Some(vr), Some(va)) = (self.b.f_c(ra.slots[0]), self.b.f_c(rb.slots[0])) {
                    (self.b.alloc_f_const(vr * va.cos())?, self.b.alloc_f_const(vr * va.sin())?)
                } else {
                    let dx = self.b.alloc_f_dyn()?;
                    let dy = self.b.alloc_f_dyn()?;
                    self.b.push_op(FxOp::CartesianF { dst_x: dx, dst_y: dy, r: ra.slots[0], ang: rb.slots[0] })?;
                    (dx, dy)
                };
                RegRef { kind: RegKind::Float, dims: 2, slots: [dx, dy, 0, 0] }
            }
            "lerp" => {
                let ra = in_ref(0)?;
                let rb = in_ref(1)?;
                let rt = in_ref(2)?;
                if ra.kind != RegKind::Float || rb.kind != RegKind::Float || rt.kind != RegKind::Float || ra.dims != rb.dims {
                    return None;
                }
                if rt.dims != ra.dims && rt.dims != 1 {
                    return None;
                }
                let mut slots = [0u16; 4];
                for k in 0..usize::from(ra.dims) {
                    let sa = *ra.slots.get(k)?;
                    let sb = *rb.slots.get(k)?;
                    let st = if rt.dims == 1 { rt.slots[0] } else { *rt.slots.get(k)? };
                    let dst = if let (Some(va), Some(vb), Some(vt)) = (self.b.f_c(sa), self.b.f_c(sb), self.b.f_c(st)) {
                        self.b.alloc_f_const(va + (vb - va) * vt)?
                    } else {
                        let dst = self.b.alloc_f_dyn()?;
                        self.b.push_op(FxOp::LerpF { dst, a: sa, b: sb, t: st })?;
                        dst
                    };
                    *slots.get_mut(k)? = dst;
                }
                RegRef { kind: RegKind::Float, dims: ra.dims, slots }
            }
            "tofloat" | "tofloat2" | "tofloat3" | "tofloat4" => {
                let ra = in_ref(0)?;
                let want = match op {
                    "tofloat" => 1u8,
                    "tofloat2" => 2,
                    "tofloat3" => 3,
                    _ => 4,
                };
                if ra.dims != want || ra.kind != RegKind::Int {
                    return None;
                }
                let mut slots = [0u16; 4];
                for k in 0..usize::from(want) {
                    let sa = *ra.slots.get(k)?;
                    let dst = if let Some(va) = self.b.i_c(sa) {
                        self.b.alloc_f_const(va as f32)?
                    } else {
                        let dst = self.b.alloc_f_dyn()?;
                        self.b.push_op(FxOp::IntToFloat { dst, a: sa })?;
                        dst
                    };
                    *slots.get_mut(k)? = dst;
                }
                RegRef { kind: RegKind::Float, dims: want, slots }
            }
            "toint1" | "toint2" | "toint3" | "toint4" => {
                is_impure = true;
                let ra = in_ref(0)?;
                let want = match op {
                    "toint1" => 1u8,
                    "toint2" => 2,
                    "toint3" => 3,
                    _ => 4,
                };
                if ra.dims != want || ra.kind != RegKind::Float {
                    return None;
                }
                let mut slots = [0u16; 4];
                for k in 0..usize::from(want) {
                    let sa = *ra.slots.get(k)?;
                    let dst = if let Some(va) = self.b.f_c(sa) {
                        if !va.is_finite() || !(-2_147_483_648.0f32..2_147_483_648.0f32).contains(&va) {
                            return None;
                        }
                        self.b.alloc_i_const(va.floor() as i32)?
                    } else {
                        let dst = self.b.alloc_i_dyn()?;
                        self.b.push_op(FxOp::FloatToInt { dst, a: sa })?;
                        dst
                    };
                    *slots.get_mut(k)? = dst;
                }
                RegRef { kind: RegKind::Int, dims: want, slots }
            }
            "eq" | "noteq" => {
                let ra = in_ref(0)?;
                let rb = in_ref(1)?;
                if ra.kind != rb.kind || ra.dims != rb.dims || ra.dims == 0 {
                    return None;
                }
                let mut eq_slot: Option<u16> = None;
                for k in 0..usize::from(ra.dims) {
                    let sa = *ra.slots.get(k)?;
                    let sb = *rb.slots.get(k)?;
                    let cmp = match ra.kind {
                        RegKind::Float => {
                            if let (Some(va), Some(vb)) = (self.b.f_c(sa), self.b.f_c(sb)) {
                                self.b.alloc_b_const(va == vb)?
                            } else {
                                let dst = self.b.alloc_b_dyn()?;
                                self.b.push_op(FxOp::EqF { dst, a: sa, b: sb })?;
                                dst
                            }
                        }
                        RegKind::Int => {
                            if let (Some(va), Some(vb)) = (self.b.i_c(sa), self.b.i_c(sb)) {
                                self.b.alloc_b_const(va == vb)?
                            } else {
                                let dst = self.b.alloc_b_dyn()?;
                                self.b.push_op(FxOp::EqI { dst, a: sa, b: sb })?;
                                dst
                            }
                        }
                        RegKind::Bool => {
                            if let (Some(va), Some(vb)) = (self.b.b_c(sa), self.b.b_c(sb)) {
                                self.b.alloc_b_const(va == vb)?
                            } else {
                                let dst = self.b.alloc_b_dyn()?;
                                self.b.push_op(FxOp::EqB { dst, a: sa, b: sb })?;
                                dst
                            }
                        }
                    };
                    eq_slot = Some(match eq_slot {
                        None => cmp,
                        Some(prev) => self.b.emit_and(prev, cmp)?,
                    });
                }
                let res = eq_slot?;
                let res = if op == "noteq" { self.b.emit_not(res)? } else { res };
                RegRef { kind: RegKind::Bool, dims: 1, slots: [res, 0, 0, 0] }
            }
            "gt" | "gteq" | "lr" | "lreq" => {
                let ra = in_ref(0)?;
                let rb = in_ref(1)?;
                if ra.kind != rb.kind || ra.dims != 1 || rb.dims != 1 {
                    return None;
                }
                let sa = ra.slots[0];
                let sb = rb.slots[0];
                let dst = match ra.kind {
                    RegKind::Float => {
                        if let (Some(va), Some(vb)) = (self.b.f_c(sa), self.b.f_c(sb)) {
                            let v = match op {
                                "gt" => va > vb,
                                "gteq" => va >= vb,
                                "lr" => va < vb,
                                _ => va <= vb,
                            };
                            self.b.alloc_b_const(v)?
                        } else {
                            let dst = self.b.alloc_b_dyn()?;
                            self.b.push_op(match op {
                                "gt" => FxOp::GtF { dst, a: sa, b: sb },
                                "gteq" => FxOp::GtEqF { dst, a: sa, b: sb },
                                "lr" => FxOp::LrF { dst, a: sa, b: sb },
                                _ => FxOp::LrEqF { dst, a: sa, b: sb },
                            })?;
                            dst
                        }
                    }
                    RegKind::Int => {
                        if let (Some(va), Some(vb)) = (self.b.i_c(sa), self.b.i_c(sb)) {
                            let v = match op {
                                "gt" => va > vb,
                                "gteq" => va >= vb,
                                "lr" => va < vb,
                                _ => va <= vb,
                            };
                            self.b.alloc_b_const(v)?
                        } else {
                            let dst = self.b.alloc_b_dyn()?;
                            self.b.push_op(match op {
                                "gt" => FxOp::GtI { dst, a: sa, b: sb },
                                "gteq" => FxOp::GtEqI { dst, a: sa, b: sb },
                                "lr" => FxOp::LrI { dst, a: sa, b: sb },
                                _ => FxOp::LrEqI { dst, a: sa, b: sb },
                            })?;
                            dst
                        }
                    }
                    RegKind::Bool => {
                        if let (Some(va), Some(vb)) = (self.b.b_c(sa), self.b.b_c(sb)) {
                            let v = match op {
                                "gt" => va & !vb,
                                "gteq" => va >= vb,
                                "lr" => !va & vb,
                                _ => va <= vb,
                            };
                            self.b.alloc_b_const(v)?
                        } else {
                            let dst = self.b.alloc_b_dyn()?;
                            self.b.push_op(match op {
                                "gt" => FxOp::GtB { dst, a: sa, b: sb },
                                "gteq" => FxOp::GtEqB { dst, a: sa, b: sb },
                                "lr" => FxOp::LrB { dst, a: sa, b: sb },
                                _ => FxOp::LrEqB { dst, a: sa, b: sb },
                            })?;
                            dst
                        }
                    }
                };
                RegRef { kind: RegKind::Bool, dims: 1, slots: [dst, 0, 0, 0] }
            }
            "and" => {
                let ra = in_ref(0)?;
                let rb = in_ref(1)?;
                if ra.kind != RegKind::Bool || rb.kind != RegKind::Bool {
                    return None;
                }
                RegRef { kind: RegKind::Bool, dims: 1, slots: [self.b.emit_and(ra.slots[0], rb.slots[0])?, 0, 0, 0] }
            }
            "or" => {
                let ra = in_ref(0)?;
                let rb = in_ref(1)?;
                if ra.kind != RegKind::Bool || rb.kind != RegKind::Bool {
                    return None;
                }
                RegRef { kind: RegKind::Bool, dims: 1, slots: [self.b.emit_or(ra.slots[0], rb.slots[0])?, 0, 0, 0] }
            }
            "not" => {
                let ra = in_ref(0)?;
                if ra.kind != RegKind::Bool {
                    return None;
                }
                RegRef { kind: RegKind::Bool, dims: 1, slots: [self.b.emit_not(ra.slots[0])?, 0, 0, 0] }
            }
            "swizzle1" | "swizzle2" | "swizzle3" | "swizzle4" | "iswizzle1" | "iswizzle2" | "iswizzle3" | "iswizzle4" => {
                let ra = in_ref(0)?;
                let want_kind = if op.starts_with('i') { RegKind::Int } else { RegKind::Float };
                if ra.kind != want_kind {
                    return None;
                }
                let count = u8::try_from(op.chars().last()?.to_digit(10)?).ok()?;
                let raw = inst.value.as_ref().map(integers).transpose().ok()?.unwrap_or([0; 4]);
                let zero = if want_kind == RegKind::Int { self.zero_i } else { self.zero_f };
                let mut slots = [0u16; 4];
                for lane in 0..usize::from(count) {
                    let c = usize::try_from(*raw.get(lane)?).ok()?;
                    if c >= 4 {
                        return None;
                    }
                    *slots.get_mut(lane)? = if c < usize::from(ra.dims) { *ra.slots.get(c)? } else { zero };
                }
                RegRef { kind: want_kind, dims: count, slots }
            }
            "vector2" | "vector3" | "vector4" | "ivector2" | "ivector3" | "ivector4" => {
                let ra = in_ref(0)?;
                let rb = in_ref(1)?;
                let want_kind = if op.starts_with('i') { RegKind::Int } else { RegKind::Float };
                let count = u8::try_from(op.chars().last()?.to_digit(10)?).ok()?;
                if ra.kind != want_kind || rb.kind != want_kind || ra.dims + rb.dims != count {
                    return None;
                }
                let mut slots = [0u16; 4];
                let da = usize::from(ra.dims);
                for k in 0..da {
                    *slots.get_mut(k)? = *ra.slots.get(k)?;
                }
                for k in 0..usize::from(rb.dims) {
                    *slots.get_mut(da + k)? = *rb.slots.get(k)?;
                }
                RegRef { kind: want_kind, dims: count, slots }
            }
            "samplelum" | "samplecol" => {
                let uv = in_ref(0)?;
                if uv.kind != RegKind::Float || uv.dims != 2 {
                    return None;
                }
                let setting = inst.value.as_ref().map(integers).transpose().ok()?.unwrap_or([0; 4]);
                let index = u16::try_from(setting[0]).ok()?;
                let bilinear = setting[1] != 0;
                if op == "samplelum" {
                    let dst = self.b.alloc_f_dyn()?;
                    self.b.push_op(FxOp::SampleLum { dst, u: uv.slots[0], v: uv.slots[1], index, bilinear })?;
                    RegRef { kind: RegKind::Float, dims: 1, slots: [dst, 0, 0, 0] }
                } else {
                    let d0 = self.b.alloc_f_dyn()?;
                    let d1 = self.b.alloc_f_dyn()?;
                    let d2 = self.b.alloc_f_dyn()?;
                    let d3 = self.b.alloc_f_dyn()?;
                    self.b.push_op(FxOp::SampleCol { dst: [d0, d1, d2, d3], u: uv.slots[0], v: uv.slots[1], index, bilinear })?;
                    RegRef { kind: RegKind::Float, dims: 4, slots: [d0, d1, d2, d3] }
                }
            }
            _ => return None,
        };

        if is_impure {
            if *branch_impure.get(idx)? {
                return None;
            }
            if in_branch > 0 {
                *branch_impure.get_mut(idx)? = true;
            }
        }

        *memo.get_mut(idx)? = Some(out);
        Some(out)
    }

    fn compile_call(
        &mut self,
        call: &crate::function::function_call::FunctionCall,
        args: &[RegRef],
        parent_scope: &Scope<'_>,
        call_depth: usize,
        in_branch: usize,
    ) -> Option<RegRef> {
        if call_depth > MAX_CALL_DEPTH {
            return None;
        }

        let mut locals: BTreeMap<String, LocalBinding> = BTreeMap::new();
        for (name, val) in &call.defaults {
            locals.insert(name.clone(), LocalBinding::Default(val.clone()));
        }
        for (name, &arg_ref) in call.inputs.iter().zip(args.iter()) {
            locals.insert(name.clone(), LocalBinding::Param(arg_ref));
        }

        // For non-exporting callees, allocate callee-local mutable slots for any `set` inside the callee
        if !call.exports {
            let mut local_sets: BTreeMap<String, u32> = BTreeMap::new();
            if !collect_local_sets(&call.program, &mut local_sets, call_depth) {
                return None;
            }
            if !local_sets.is_empty() {
                let entry_scope = Scope { parent: Some(parent_scope), locals: locals.clone(), exporting: false };
                for (name, tc) in local_sets {
                    let (kind, dims) = reg_from_type_code(tc)?;
                    let (init_ref, _) = self.resolve_variable(&entry_scope, "get_", &name, tc)?;
                    let mut_slot = self.b.alloc_dyn_ref(kind, dims)?;
                    self.b.emit_copy(mut_slot, init_ref)?;
                    locals.insert(name, LocalBinding::Mutable(mut_slot));
                }
            }
        }

        let callee_scope = Scope { parent: Some(parent_scope), locals, exporting: call.exports };
        self.compile_program(&call.program, &callee_scope, call_depth, in_branch)
    }
}

fn collect_local_sets(program: &Program, sets: &mut BTreeMap<String, u32>, depth: usize) -> bool {
    if depth > MAX_CALL_DEPTH {
        return false;
    }
    for inst in &program.instructions {
        if let Some(call) = &inst.call {
            if call.exports && !collect_local_sets(&call.program, sets, depth + 1) {
                return false;
            }
            continue;
        }
        if inst.definition.id == "set" {
            let Some(Value::Text(name)) = &inst.value else { return false };
            if name.is_empty() || name.starts_with('$') {
                return false;
            }
            if let Some(&prev) = sets.get(name) {
                if prev != inst.type_code {
                    return false;
                }
            } else {
                sets.insert(name.clone(), inst.type_code);
            }
        }
    }
    true
}

fn program_has_impure_effects(program: &Program, schema: &FxVarSchema, scope: &Scope<'_>, depth: usize) -> bool {
    if depth > MAX_CALL_DEPTH {
        return true;
    }
    for inst in &program.instructions {
        if let Some(call) = &inst.call {
            if program_has_impure_effects(&call.program, schema, scope, depth + 1) {
                return true;
            }
            continue;
        }
        match inst.definition.id {
            "rand" | "set" | "while" | "isdefined" | "toint1" | "toint2" | "toint3" | "toint4" => return true,
            "add" | "sub" | "mul" | "div" | "mod" | "neg" | "abs" if matches!(inst.type_code, 16 | 32 | 64 | 128) => return true,
            op if op.starts_with("get_") => {
                if let Some(Value::Text(name)) = &inst.value
                    && (schema.by_name.contains_key(name) || matches!(scope.lookup(name), Some(LocalBinding::Mutable(_))))
                {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

impl FxProgram {
    pub(super) fn try_compile(program: &Program, vars: &BTreeMap<String, Value>, size: [f32; 2], tile: bool, schema: &FxVarSchema) -> Option<Self> {
        if !schema.is_valid() {
            return None;
        }

        let mut b = Builder::new();
        let pos_x = b.alloc_f_dyn()?;
        let pos_y = b.alloc_f_dyn()?;
        let depth = b.alloc_f_dyn()?;
        let extent = b.alloc_f_dyn()?;
        let number = b.alloc_f_dyn()?;
        let seed = b.alloc_i_dyn()?;
        let zero_f = b.alloc_f_const(0.0)?;
        let zero_i = b.alloc_i_const(0)?;

        let mut env = CompileEnv {
            b,
            schema,
            vars,
            dummy_ctx: EvaluationContext { scope: FunctionScope::FxMap, position: [0.0, 0.0], size, images: &[], tile, variables: vars },
            pos_ref: RegRef { kind: RegKind::Float, dims: 2, slots: [pos_x, pos_y, 0, 0] },
            depth_ref: RegRef { kind: RegKind::Float, dims: 1, slots: [depth, 0, 0, 0] },
            extent_ref: RegRef { kind: RegKind::Float, dims: 1, slots: [extent, 0, 0, 0] },
            number_ref: RegRef { kind: RegKind::Float, dims: 1, slots: [number, 0, 0, 0] },
            seed_ref: RegRef { kind: RegKind::Int, dims: 1, slots: [seed, 0, 0, 0] },
            zero_f,
            zero_i,
        };

        let root_scope = Scope { parent: None, locals: BTreeMap::new(), exporting: true };
        let output = env.compile_program(program, &root_scope, 0, 0)?;

        if output.kind == RegKind::Int && output.dims != 1 {
            return None;
        }

        Some(Self { ops: env.b.ops, f_init: env.b.f_init, i_init: env.b.i_init, b_init: env.b.b_init, pos_x, pos_y, depth, extent, number, seed, output })
    }

    pub(super) fn eval(
        &self,
        regs: &mut FxRegs,
        vars: &mut FxVarStore,
        inputs: &FxVisitInputs,
        random: &mut u32,
        images: &[Option<craft_texture::Image>],
        tile: bool,
    ) -> Result<FxOutput, String> {
        regs.f.clear();
        regs.f.extend_from_slice(&self.f_init);
        regs.i.clear();
        regs.i.extend_from_slice(&self.i_init);
        regs.b.clear();
        regs.b.extend_from_slice(&self.b_init);

        let rf = &mut regs.f;
        let ri = &mut regs.i;
        let rb = &mut regs.b;
        let vf = &mut vars.f;
        let vi = &mut vars.i;
        let vb = &mut vars.b;

        if let Some(s) = rf.get_mut(usize::from(self.pos_x)) {
            *s = inputs.pos[0];
        }
        if let Some(s) = rf.get_mut(usize::from(self.pos_y)) {
            *s = inputs.pos[1];
        }
        if let Some(s) = rf.get_mut(usize::from(self.depth)) {
            *s = inputs.depth;
        }
        if let Some(s) = rf.get_mut(usize::from(self.extent)) {
            *s = inputs.extent;
        }
        if let Some(s) = rf.get_mut(usize::from(self.number)) {
            *s = inputs.number;
        }
        if let Some(s) = ri.get_mut(usize::from(self.seed)) {
            *s = inputs.seed;
        }

        let ops = &self.ops;
        let mut pc: usize = 0;
        while let Some(&op) = ops.get(pc) {
            pc += 1;
            match op {
                FxOp::CopyF { dst, src } => rf[usize::from(dst)] = rf[usize::from(src)],
                FxOp::CopyI { dst, src } => ri[usize::from(dst)] = ri[usize::from(src)],
                FxOp::CopyB { dst, src } => rb[usize::from(dst)] = rb[usize::from(src)],
                FxOp::LoadVarF { dst, var } => rf[usize::from(dst)] = vf[usize::from(var)],
                FxOp::LoadVarI { dst, var } => ri[usize::from(dst)] = vi[usize::from(var)],
                FxOp::LoadVarB { dst, var } => rb[usize::from(dst)] = vb[usize::from(var)],
                FxOp::StoreVarF { var, src } => vf[usize::from(var)] = rf[usize::from(src)],
                FxOp::StoreVarI { var, src } => vi[usize::from(var)] = ri[usize::from(src)],
                FxOp::StoreVarB { var, src } => vb[usize::from(var)] = rb[usize::from(src)],
                FxOp::JumpIfFalse { cond, target } => {
                    if !rb[usize::from(cond)] {
                        pc = target as usize;
                    }
                }
                FxOp::Jump { target } => {
                    pc = target as usize;
                }
                FxOp::NegF { dst, a } => rf[usize::from(dst)] = -rf[usize::from(a)],
                FxOp::AbsF { dst, a } => rf[usize::from(dst)] = rf[usize::from(a)].abs(),
                FxOp::FloorF { dst, a } => rf[usize::from(dst)] = rf[usize::from(a)].floor(),
                FxOp::CeilF { dst, a } => rf[usize::from(dst)] = rf[usize::from(a)].ceil(),
                FxOp::SqrtF { dst, a } => rf[usize::from(dst)] = rf[usize::from(a)].sqrt(),
                FxOp::ExpF { dst, a } => rf[usize::from(dst)] = rf[usize::from(a)].exp(),
                FxOp::LogF { dst, a } => rf[usize::from(dst)] = rf[usize::from(a)].ln(),
                FxOp::Log2F { dst, a } => rf[usize::from(dst)] = rf[usize::from(a)].log2(),
                FxOp::Pow2F { dst, a } => rf[usize::from(dst)] = rf[usize::from(a)].exp2(),
                FxOp::SinF { dst, a } => rf[usize::from(dst)] = rf[usize::from(a)].sin(),
                FxOp::CosF { dst, a } => rf[usize::from(dst)] = rf[usize::from(a)].cos(),
                FxOp::TanF { dst, a } => rf[usize::from(dst)] = rf[usize::from(a)].tan(),
                FxOp::RandF { dst, a } => rf[usize::from(dst)] = crate::random::pattern_function(random) * rf[usize::from(a)],
                FxOp::AddF { dst, a, b } => rf[usize::from(dst)] = rf[usize::from(a)] + rf[usize::from(b)],
                FxOp::SubF { dst, a, b } => rf[usize::from(dst)] = rf[usize::from(a)] - rf[usize::from(b)],
                FxOp::MulF { dst, a, b } => rf[usize::from(dst)] = rf[usize::from(a)] * rf[usize::from(b)],
                FxOp::DivF { dst, a, b } => rf[usize::from(dst)] = rf[usize::from(a)] / rf[usize::from(b)],
                FxOp::ModF { dst, a, b } => {
                    let va = rf[usize::from(a)];
                    let vb = rf[usize::from(b)];
                    rf[usize::from(dst)] = va - vb * (va / vb).floor();
                }
                FxOp::MinF { dst, a, b } => rf[usize::from(dst)] = rf[usize::from(a)].min(rf[usize::from(b)]),
                FxOp::MaxF { dst, a, b } => rf[usize::from(dst)] = rf[usize::from(a)].max(rf[usize::from(b)]),
                FxOp::PowF { dst, a, b } => rf[usize::from(dst)] = rf[usize::from(a)].powf(rf[usize::from(b)]),
                FxOp::Atan2F { dst, y, x } => rf[usize::from(dst)] = rf[usize::from(y)].atan2(rf[usize::from(x)]),
                FxOp::CartesianF { dst_x, dst_y, r, ang } => {
                    let vr = rf[usize::from(r)];
                    let (s, c) = rf[usize::from(ang)].sin_cos();
                    rf[usize::from(dst_x)] = vr * c;
                    rf[usize::from(dst_y)] = vr * s;
                }
                FxOp::LerpF { dst, a, b, t } => {
                    let va = rf[usize::from(a)];
                    let vb = rf[usize::from(b)];
                    rf[usize::from(dst)] = va + (vb - va) * rf[usize::from(t)];
                }
                FxOp::AddI { dst, a, b } => {
                    ri[usize::from(dst)] = ri[usize::from(a)].checked_add(ri[usize::from(b)]).ok_or("add: integer overflow or division by zero")?;
                }
                FxOp::SubI { dst, a, b } => {
                    ri[usize::from(dst)] = ri[usize::from(a)].checked_sub(ri[usize::from(b)]).ok_or("sub: integer overflow or division by zero")?;
                }
                FxOp::MulI { dst, a, b } => {
                    ri[usize::from(dst)] = ri[usize::from(a)].checked_mul(ri[usize::from(b)]).ok_or("mul: integer overflow or division by zero")?;
                }
                FxOp::DivI { dst, a, b } => {
                    ri[usize::from(dst)] = ri[usize::from(a)].checked_div(ri[usize::from(b)]).ok_or("div: integer overflow or division by zero")?;
                }
                FxOp::ModI { dst, a, b } => {
                    ri[usize::from(dst)] = ri[usize::from(a)].checked_rem(ri[usize::from(b)]).ok_or("mod: integer overflow or division by zero")?;
                }
                FxOp::NegI { dst, a } => {
                    ri[usize::from(dst)] = ri[usize::from(a)].checked_neg().ok_or("neg: integer overflow or division by zero")?;
                }
                FxOp::AbsI { dst, a } => {
                    ri[usize::from(dst)] = ri[usize::from(a)].checked_abs().ok_or("abs: integer overflow or division by zero")?;
                }
                FxOp::MinI { dst, a, b } => ri[usize::from(dst)] = ri[usize::from(a)].min(ri[usize::from(b)]),
                FxOp::MaxI { dst, a, b } => ri[usize::from(dst)] = ri[usize::from(a)].max(ri[usize::from(b)]),
                FxOp::IntToFloat { dst, a } => rf[usize::from(dst)] = ri[usize::from(a)] as f32,
                FxOp::FloatToInt { dst, a } => {
                    let v = rf[usize::from(a)];
                    if !v.is_finite() || !(-2_147_483_648.0f32..2_147_483_648.0f32).contains(&v) {
                        return Err("cannot cast non-finite or out-of-range float to integer".into());
                    }
                    ri[usize::from(dst)] = v.floor() as i32;
                }
                FxOp::IntToBool { dst, a } => rb[usize::from(dst)] = ri[usize::from(a)] != 0,
                FxOp::EqF { dst, a, b } => rb[usize::from(dst)] = rf[usize::from(a)] == rf[usize::from(b)],
                FxOp::EqI { dst, a, b } => rb[usize::from(dst)] = ri[usize::from(a)] == ri[usize::from(b)],
                FxOp::EqB { dst, a, b } => rb[usize::from(dst)] = rb[usize::from(a)] == rb[usize::from(b)],
                FxOp::GtF { dst, a, b } => rb[usize::from(dst)] = rf[usize::from(a)] > rf[usize::from(b)],
                FxOp::GtEqF { dst, a, b } => rb[usize::from(dst)] = rf[usize::from(a)] >= rf[usize::from(b)],
                FxOp::LrF { dst, a, b } => rb[usize::from(dst)] = rf[usize::from(a)] < rf[usize::from(b)],
                FxOp::LrEqF { dst, a, b } => rb[usize::from(dst)] = rf[usize::from(a)] <= rf[usize::from(b)],
                FxOp::GtI { dst, a, b } => rb[usize::from(dst)] = ri[usize::from(a)] > ri[usize::from(b)],
                FxOp::GtEqI { dst, a, b } => rb[usize::from(dst)] = ri[usize::from(a)] >= ri[usize::from(b)],
                FxOp::LrI { dst, a, b } => rb[usize::from(dst)] = ri[usize::from(a)] < ri[usize::from(b)],
                FxOp::LrEqI { dst, a, b } => rb[usize::from(dst)] = ri[usize::from(a)] <= ri[usize::from(b)],
                FxOp::GtB { dst, a, b } => rb[usize::from(dst)] = rb[usize::from(a)] & !rb[usize::from(b)],
                FxOp::GtEqB { dst, a, b } => rb[usize::from(dst)] = rb[usize::from(a)] >= rb[usize::from(b)],
                FxOp::LrB { dst, a, b } => rb[usize::from(dst)] = !rb[usize::from(a)] & rb[usize::from(b)],
                FxOp::LrEqB { dst, a, b } => rb[usize::from(dst)] = rb[usize::from(a)] <= rb[usize::from(b)],
                FxOp::And { dst, a, b } => rb[usize::from(dst)] = rb[usize::from(a)] && rb[usize::from(b)],
                FxOp::Or { dst, a, b } => rb[usize::from(dst)] = rb[usize::from(a)] || rb[usize::from(b)],
                FxOp::Not { dst, a } => rb[usize::from(dst)] = !rb[usize::from(a)],
                FxOp::SampleLum { dst, u, v, index, bilinear } => {
                    let vu = rf[usize::from(u)];
                    let vv = rf[usize::from(v)];
                    let val = match images.get(usize::from(index)).and_then(Option::as_ref) {
                        Some(image) if !bilinear => image.get((vu * image.width() as f32).floor() as i64, (vv * image.height() as f32).floor() as i64, tile)[0],
                        Some(image) => image.sample_stored(vu, vv, tile)[0],
                        None => 0.0,
                    };
                    rf[usize::from(dst)] = val;
                }
                FxOp::SampleCol { dst, u, v, index, bilinear } => {
                    let vu = rf[usize::from(u)];
                    let vv = rf[usize::from(v)];
                    let px = match images.get(usize::from(index)).and_then(Option::as_ref) {
                        Some(image) if !bilinear => image.get((vu * image.width() as f32).floor() as i64, (vv * image.height() as f32).floor() as i64, tile),
                        Some(image) => image.sample_stored(vu, vv, tile),
                        None => [0.0; 4],
                    };
                    rf[usize::from(dst[0])] = px[0];
                    rf[usize::from(dst[1])] = px[1];
                    rf[usize::from(dst[2])] = px[2];
                    rf[usize::from(dst[3])] = px[3];
                }
            }
        }

        let out = match (self.output.kind, self.output.dims) {
            (RegKind::Bool, _) => FxOutput::Bool(rb[usize::from(self.output.slots[0])]),
            (RegKind::Int, _) => FxOutput::Int(ri[usize::from(self.output.slots[0])]),
            (RegKind::Float, 1) => {
                let v = rf[usize::from(self.output.slots[0])];
                if !v.is_finite() {
                    return Err("function produced a non-finite output".into());
                }
                FxOutput::Float(v)
            }
            (RegKind::Float, 2) => {
                let v = [rf[usize::from(self.output.slots[0])], rf[usize::from(self.output.slots[1])]];
                if !v[0].is_finite() || !v[1].is_finite() {
                    return Err("function produced a non-finite output".into());
                }
                FxOutput::Float2(v)
            }
            (RegKind::Float, 3) => {
                let v = [rf[usize::from(self.output.slots[0])], rf[usize::from(self.output.slots[1])], rf[usize::from(self.output.slots[2])]];
                if v.iter().any(|c| !c.is_finite()) {
                    return Err("function produced a non-finite output".into());
                }
                FxOutput::Float3(v)
            }
            (RegKind::Float, _) => {
                let v = [
                    rf[usize::from(self.output.slots[0])],
                    rf[usize::from(self.output.slots[1])],
                    rf[usize::from(self.output.slots[2])],
                    rf[usize::from(self.output.slots[3])],
                ];
                if v.iter().any(|c| !c.is_finite()) {
                    return Err("function produced a non-finite output".into());
                }
                FxOutput::Float4(v)
            }
        };

        Ok(out)
    }
}

fn reg_type_code(kind: RegKind, dims: u8) -> u32 {
    match (kind, dims) {
        (RegKind::Bool, _) => ValueType::Boolean as u32,
        (RegKind::Int, 1) => 16,
        (RegKind::Int, 2) => 32,
        (RegKind::Int, 3) => 64,
        (RegKind::Int, 4) => 128,
        (RegKind::Float, 1) => 256,
        (RegKind::Float, 2) => 512,
        (RegKind::Float, 3) => 1024,
        (RegKind::Float, 4) => 2048,
        _ => 0,
    }
}

fn reg_from_type_code(code: u32) -> Option<(RegKind, u8)> {
    match code {
        4 => Some((RegKind::Bool, 1)),
        16 => Some((RegKind::Int, 1)),
        32 => Some((RegKind::Int, 2)),
        64 => Some((RegKind::Int, 3)),
        128 => Some((RegKind::Int, 4)),
        256 => Some((RegKind::Float, 1)),
        512 => Some((RegKind::Float, 2)),
        1024 => Some((RegKind::Float, 3)),
        2048 => Some((RegKind::Float, 4)),
        _ => None,
    }
}
