# SPDX-License-Identifier: GPL-2.0-or-later
"""
Gushing Over Blender FX Addon
=============================
A cohesive, single-file add-on suite for Blender 4.2 LTS / 5.2 LTS.
Integrates multiple robust visual enhancements into one centralized tool:
1. Magia Baiser FX (Procedural GPU feedback effects)
2. UI Text Outline (Readable theme interface text)
3. Halftone Viewport Overlay (Stylized manga screen-tones)
4. Magical Girls Chaos (Theme animator & GPU silhouettes)

Architecture Note:
To maintain single-file portability while ensuring professional code hygiene, 
global variables are strictly avoided. All cross-tick state is encapsulated in 
domain-specific container classes (GBFX_State, MB_State, HT_State, GSH_State).
"""

bl_info = {
    "name": "Gushing Over Blender FX Addon",
    "author": "Grizlik",
    "version": (1, 4, 3),
    "blender": (4, 2, 0),
    "location": "3D Viewport > Sidebar (N) > Gushing FX",
    "description": "Unified visual FX suite: GPU silhouettes, halftone overlays, UI text outlines, and procedural feedback effects.",
    "category": "Interface",
}

import bpy
import bmesh
import gpu
import math
import random
import time
import os
import xml.etree.ElementTree as ET
import traceback
import blf
from contextlib import contextmanager

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

from gpu_extras.batch import batch_for_shader
from bpy_extras import view3d_utils
from mathutils import Vector, Matrix, Euler
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, StringProperty
from bpy.types import AddonPreferences, Operator, Panel
from bpy.app.handlers import persistent

ADDON_ID = __package__ if __package__ else __name__

# =============================================================================
# STATE ENCAPSULATION & UTILITIES
# =============================================================================

class GBFX_State:
    def __init__(self):
        self.errors = []
    
    def log_error(self, context_str, exc):
        """Logs exceptions to the console and to the UI for user visibility."""
        msg = f"{context_str}: {exc}"
        prefs = get_prefs()
        if prefs and prefs.gbfx_debug_logging:
            print(f"[Gushing FX Error] {msg}")
            traceback.print_exc()
            
        if msg not in self.errors:
            self.errors.insert(0, msg)
            if len(self.errors) > 5:
                self.errors.pop()
        gbfx_tag_redraw({'VIEW_3D'})

gbfx_state = GBFX_State()

class MB_State:
    def __init__(self):
        self.running = False
        self.effects = []
        self.obj_count = 0
        self.obj_cache = {}
        self.suppress_until = 0.0
        self.last_modifier_op_id = None
        self.last_selection_hash = None
        self.msgbus_owner = object()
        self.handle_view = None
        self.handle_pixel = None
        self.star_cache = {}
        self.flower_cache = {}
        self.shader = None

mb_state = MB_State()

class HT_State:
    def __init__(self):
        self.shader = None
        self.batch = None
        self.handle = None

ht_state = HT_State()

class GSH_State:
    def __init__(self):
        self.master_timer_running = False
        self.chaos_active = False
        self.chaos_rot = 0.0
        self.outline_rot = 0.0
        self.border_rot = 0.0
        self.chaos_targets = []
        
        self.vp_handle = None
        self.node_handle = None
        self.border_handle = None
        
        self.depth_shader = None
        self.mask_shader = None
        self.compose_shader = None
        self.outline_offscreens = {}
        self.batch_cache = {}
        self.bone_batch_cache = {}
        # Objects whose cached batch is known stale (geometry changed) but a
        # rebuild hasn't happened yet. Kept separate from batch_cache so the
        # last-known-good batch can still be drawn (throttled) instead of the
        # rebuild happening synchronously on every single depsgraph tick.
        self.batch_dirty = set()
        self.batch_last_build = {}

gsh_state = GSH_State()


def get_prefs(context=None):
    ctx = context or bpy.context
    if not ctx or not hasattr(ctx, "preferences"): return None
    addon = ctx.preferences.addons.get(ADDON_ID)
    if addon and hasattr(addon, "preferences"): return addon.preferences
    # Fallback
    for item in ctx.preferences.addons:
        if hasattr(item, "preferences") and hasattr(item.preferences, "mb_enable_fx"):
            return item.preferences
    return None

def gbfx_tag_redraw(area_types={'VIEW_3D'}):
    """Centralized utility to avoid nested loop duplication."""
    try:
        # During Addon registration, bpy.data is restricted. 
        # This try/except safely bypasses redraws during the boot phase.
        for wm in bpy.data.window_managers:
            for w in wm.windows:
                if w.screen:
                    for a in w.screen.areas:
                        if a.type in area_types:
                            a.tag_redraw()
    except Exception:
        pass

def hex_rgba(hex_str, alpha=1.0):
    """Raw sRGB conversion for exact viewport matching in POST_PIXEL."""
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 6: hex_str += "ff"
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    a = int(hex_str[6:8], 16) / 255.0 if len(hex_str) > 6 else alpha
    return (r, g, b, a)

def bbox_center_and_size(ob, depsgraph=None):
    ob_eval = ob
    if depsgraph is not None:
        try: ob_eval = ob.evaluated_get(depsgraph)
        except Exception: ob_eval = ob
    try: corners = [ob_eval.matrix_world @ Vector(c) for c in ob_eval.bound_box]
    except Exception: corners = []
    
    if not corners:
        return ob.matrix_world.translation.copy(), Vector((0.2, 0.2, 0.2))
    
    xs, ys, zs = [c.x for c in corners], [c.y for c in corners], [c.z for c in corners]
    center = Vector(((min(xs)+max(xs))*0.5, (min(ys)+max(ys))*0.5, (min(zs)+max(zs))*0.5))
    size = Vector((max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)))
    if size.length < 1e-5: size = Vector((0.2, 0.2, 0.2))
    return center, size

def _current_depsgraph():
    try: return bpy.context.evaluated_depsgraph_get()
    except Exception: return None

def world_to_region(context, world_pos):
    region, rv3d = context.region, context.region_data
    if not region or not rv3d: return None
    return view3d_utils.location_3d_to_region_2d(region, rv3d, world_pos)


# =============================================================================
# UNIFIED PREFERENCES & COLOR PALETTE
# =============================================================================

def _gbfx_palette_update(self, context):
    gbfx_tag_redraw({'VIEW_3D', 'NODE_EDITOR', 'IMAGE_EDITOR', 'UI', 'PROPERTIES', 'OUTLINER'})

def gbfx_colors(prefs):
    """Returns the 4 main accent colors."""
    return [prefs.gbfx_color_1, prefs.gbfx_color_2, prefs.gbfx_color_3, prefs.gbfx_color_4]

def _uto_update_and_apply(self, context):
    uto_apply_outline(context)

def _ht_on_setting_changed(self, context):
    gbfx_tag_redraw({'VIEW_3D'})

_chaos_debounce_timer = None
def _trigger_chaos_rescan():
    gsh_rescan_if_enabled()
    return None

def _gsh_on_base_color_update(self, context):
    """Debounced update to prevent theme-walk stutter while dragging the color picker."""
    global _chaos_debounce_timer
    if bpy.app.timers.is_registered(_trigger_chaos_rescan):
        bpy.app.timers.unregister(_trigger_chaos_rescan)
    bpy.app.timers.register(_trigger_chaos_rescan, first_interval=0.3)

def _gsh_evaluate_state_cb(self, context):
    gsh_evaluate_state(context)

class GBFX_Preferences(AddonPreferences):
    bl_idname = ADDON_ID

    gbfx_debug_logging: BoolProperty(name="Verbose Console Logging", default=False, description="Print caught errors to console")

    # --- 0. GLOBAL COLOR PALETTE ---
    # Used COLOR_GAMMA subtype so that Blender's UI color picker treats the stored value as sRGB.
    # This ensures colors match exactly what the user picked when drawn via UNIFORM_COLOR.
    gbfx_color_1: FloatVectorProperty(name="Color 1 (Slash/Error)", subtype="COLOR_GAMMA", size=4, min=0.0, max=1.0, default=hex_rgba('ff0092'), update=_gbfx_palette_update)
    gbfx_color_2: FloatVectorProperty(name="Color 2 (Grasp/Save)", subtype="COLOR_GAMMA", size=4, min=0.0, max=1.0, default=hex_rgba('ffd700'), update=_gbfx_palette_update)
    gbfx_color_3: FloatVectorProperty(name="Color 3 (Select)", subtype="COLOR_GAMMA", size=4, min=0.0, max=1.0, default=hex_rgba('1f89c7'), update=_gbfx_palette_update)
    gbfx_color_4: FloatVectorProperty(name="Color 4 (Add/Undo)", subtype="COLOR_GAMMA", size=4, min=0.0, max=1.0, default=hex_rgba('7eca2c'), update=_gbfx_palette_update)
    gbfx_color_dark: FloatVectorProperty(name="Dark Accent (Dark Energy)", subtype="COLOR_GAMMA", size=4, min=0.0, max=1.0, default=hex_rgba('161123'), update=_gbfx_palette_update)

    # --- 1. MAGIA BAISER FX SETTINGS ---
    mb_enable_fx: BoolProperty(name="Enable Effects", default=True)
    mb_trig_delete: BoolProperty(name="Delete — Baiser's Slash", default=True)
    mb_trig_add: BoolProperty(name="Add — Alice's Toybox", default=True)
    mb_trig_select: BoolProperty(name="Selection — Azure's Ripple", default=True)
    mb_trig_apply_modifier: BoolProperty(name="Apply Modifier — Enormita's Grasp", default=True)
    mb_trig_undo: BoolProperty(name="Undo — Alice's Rewind", default=True)
    mb_trig_redo: BoolProperty(name="Redo — Sulfur's Forward", default=True)
    mb_trig_error: BoolProperty(name="Error (Via Apply Modifiers FX)", default=True, description="Triggers ONLY from the 'Apply All Modifiers' button or Preview")
    mb_trig_save: BoolProperty(name="Save — Enormita Salute", default=True)
    mb_style_preset: EnumProperty(items=[('MAGIA_BAISER', "Magia Baiser", ""), ('TRES_MAGIA', "Tres Magia", "")], default='MAGIA_BAISER')
    mb_intensity: FloatProperty(name="Gushing Intensity", default=1.0, min=0.0, max=2.0, subtype='FACTOR')
    mb_selection_tracking_mode: EnumProperty(items=[('ACTIVE_ONLY', "Active Object Only", ""), ('FULL_SELECTION', "Full Selection", "")], default='ACTIVE_ONLY')
    mb_preview_effect: EnumProperty(items=[('DELETE', "Delete", ""), ('ADD', "Add", ""), ('SELECT', "Select", ""), ('APPLY_MODIFIER', "Apply Mod", ""), ('UNDO', "Undo", ""), ('REDO', "Redo", ""), ('ERROR', "Error", ""), ('SAVE', "Save", "")], default='DELETE')
    mb_show_advanced: BoolProperty(name="Show Triggers", default=False)

    # --- 2. UI TEXT OUTLINE SETTINGS ---
    uto_enabled: BoolProperty(name="Enable Text Outline", default=True, update=_uto_update_and_apply)
    uto_outline_style: EnumProperty(items=(("SOFT", "Soft Halo", ""), ("WIDE", "Wide Halo", ""), ("CRISP", "Crisp Outline", "")), default="CRISP", update=_uto_update_and_apply)
    uto_intensity: FloatProperty(name="Intensity", default=0.85, min=0.0, max=1.0, subtype="FACTOR", update=_uto_update_and_apply)
    uto_color_mode: EnumProperty(items=(("DARK", "Dark Outline", ""), ("LIGHT", "Light Outline", "")), default="DARK", update=_uto_update_and_apply)
    uto_apply_widget: BoolProperty(name="Buttons & Fields", default=True, update=_uto_update_and_apply)
    uto_apply_widget_label: BoolProperty(name="Labels", default=True, update=_uto_update_and_apply)
    uto_apply_panel_title: BoolProperty(name="Panel & Header Titles", default=True, update=_uto_update_and_apply)
    uto_apply_tooltip: BoolProperty(name="Tooltips", default=True, update=_uto_update_and_apply)
    uto_show_advanced: BoolProperty(name="Show Advanced Options", default=False)

    # --- 3. HALFTONE BG SETTINGS ---
    ht_enabled: BoolProperty(name="Enable Halftone Overlay", default=False, update=_ht_on_setting_changed)
    ht_dot_color: FloatVectorProperty(name="Dot Color", subtype='COLOR_GAMMA', size=4, default=(0.05, 0.02, 0.10, 0.85), min=0.0, max=1.0, update=_ht_on_setting_changed)
    ht_dot_density: FloatProperty(name="Dot Density", default=0.06, min=0.01, max=0.5, precision=3, update=_ht_on_setting_changed)
    ht_max_dot_size: FloatProperty(name="Max Size (Shadow)", default=0.44, min=0.0, max=0.5, precision=3, update=_ht_on_setting_changed)
    ht_min_dot_size: FloatProperty(name="Min Size (Light)", default=0.0, min=0.0, max=0.5, precision=3, update=_ht_on_setting_changed)
    ht_gradient_direction: EnumProperty(items=(('VERTICAL', "Vertical", ""), ('HORIZONTAL', "Horizontal", ""), ('DIAGONAL', "Diagonal", ""), ('VIGNETTE', "Vignette", "")), default='VIGNETTE', update=_ht_on_setting_changed)
    ht_mask_coverage: FloatProperty(name="Coverage / Size", default=1.0, min=0.0, max=2.5, precision=2, update=_ht_on_setting_changed)
    ht_opacity: FloatProperty(name="Opacity", default=1.0, min=0.0, max=1.0, subtype='FACTOR', update=_ht_on_setting_changed)
    ht_pattern_rotation: FloatProperty(name="Pattern Rotation", default=math.radians(45.0), subtype='ANGLE', unit='ROTATION', update=_ht_on_setting_changed)
    ht_show_advanced: BoolProperty(name="Show Details", default=False)

    # --- 4. GUSHING CHAOS SETTINGS ---
    gsh_chaos_enabled: BoolProperty(name="Enable Theme Animator", default=False, update=_gsh_evaluate_state_cb)
    gsh_chaos_speed: FloatProperty(name="Rotation Period (s)", default=1.5, min=0.3, max=10.0)
    gsh_chaos_base_color: FloatVectorProperty(name="Target Theme Color", subtype="COLOR_GAMMA", size=4, min=0.0, max=1.0, default=hex_rgba('be6400'), update=_gsh_on_base_color_update)
    gsh_viewport_outline_enabled: BoolProperty(name="Enable Viewport Silhouette Outline", default=False, update=_gsh_evaluate_state_cb)
    gsh_node_outline_enabled: BoolProperty(name="Enable Node Editor Active Outline", default=False, update=_gsh_evaluate_state_cb)
    gsh_viewport_outline_width: FloatProperty(name="Outline Width (px)", default=4.0, min=1.0, max=24.0)
    gsh_viewport_outline_resolution: FloatProperty(name="Mask Resolution Scale", default=0.5, min=0.25, max=1.0)
    gsh_viewport_outline_speed: FloatProperty(name="Rotation Period (s)", default=1.5, min=0.15, max=10.0)
    gsh_viewport_outline_depth_test: BoolProperty(name="Occlude By Scene Geometry", default=True)
    gsh_viewport_outline_edit_mode: BoolProperty(name="Outline Edit-Mode Selection", default=True)
    gsh_border_enabled: BoolProperty(name="Enable Activity Border", default=False, update=_gsh_evaluate_state_cb)
    gsh_border_speed: FloatProperty(name="Rotation Period (s)", default=1.0, min=0.15, max=10.0)
    gsh_border_line_width: FloatProperty(name="Outline Thickness (px)", default=4.0, min=1.0, max=20.0)
    gsh_show_advanced: BoolProperty(name="Show Advanced Options", default=False)

    def _draw_palette(self, layout):
        box = layout.box()
        box.label(text="Global Color Palette", icon='COLOR')
        grid = box.grid_flow(columns=2, even_columns=True)
        grid.prop(self, "gbfx_color_1")
        grid.prop(self, "gbfx_color_2")
        grid.prop(self, "gbfx_color_3")
        grid.prop(self, "gbfx_color_4")
        box.prop(self, "gbfx_color_dark")

    def _draw_magia_baiser(self, layout):
        box = layout.box()
        box.label(text="1. Magia Baiser FX", icon='SHADERFX')
        box.prop(self, "mb_enable_fx", toggle=True, icon='PLAY')
        if self.mb_enable_fx:
            col = box.column()
            row = col.row()
            row.prop(self, "mb_show_advanced", icon="TRIA_DOWN" if self.mb_show_advanced else "TRIA_RIGHT", emboss=False)
            if self.mb_show_advanced:
                b = col.box()
                b.label(text="Triggers", icon='OPTIONS')
                grid = b.grid_flow(columns=2, even_columns=True)
                for p in ("mb_trig_delete", "mb_trig_add", "mb_trig_select", "mb_trig_apply_modifier", "mb_trig_undo", "mb_trig_redo", "mb_trig_error", "mb_trig_save"):
                    grid.prop(self, p)
            b = col.box()
            b.label(text="Style & Tracking", icon='COLOR')
            b.prop(self, "mb_style_preset")
            b.prop(self, "mb_intensity")
            b.prop(self, "mb_selection_tracking_mode", expand=True)

    def _draw_text_outline(self, layout):
        box = layout.box()
        box.label(text="2. UI Text Outline", icon='USER')
        box.prop(self, "uto_enabled", text="Enable Text Outline", toggle=True)
        if self.uto_enabled:
            body = box.column()
            body.prop(self, "uto_outline_style")
            body.prop(self, "uto_intensity", slider=True)
            body.prop(self, "uto_color_mode")
            adv = box.box()
            row = adv.row()
            row.prop(self, "uto_show_advanced", icon="TRIA_DOWN" if self.uto_show_advanced else "TRIA_RIGHT", emboss=False)
            if self.uto_show_advanced:
                adv_body = adv.column()
                grid = adv_body.grid_flow(columns=2, even_columns=True)
                grid.prop(self, "uto_apply_widget")
                grid.prop(self, "uto_apply_widget_label")
                grid.prop(self, "uto_apply_panel_title")
                grid.prop(self, "uto_apply_tooltip")
            box.operator("gbfx.uto_refresh", icon="FILE_REFRESH", text="Refresh Now")

    def _draw_halftone(self, layout):
        box = layout.box()
        box.label(text="3. Halftone Overlay", icon='SHADING_RENDERED')
        box.prop(self, "ht_enabled", text="Enable Halftone", toggle=True)
        if self.ht_enabled:
            col = box.column()
            b = col.box()
            b.label(text="Appearance", icon='COLOR')
            b.prop(self, "ht_dot_color")
            b.prop(self, "ht_opacity")
            row = col.row()
            row.prop(self, "ht_show_advanced", icon="TRIA_DOWN" if self.ht_show_advanced else "TRIA_RIGHT", emboss=False, text="Mask & Pattern Settings")
            if self.ht_show_advanced:
                b = col.box()
                b.prop(self, "ht_gradient_direction", text="Shape")
                b.prop(self, "ht_mask_coverage", text="Coverage / Size")
                b = col.box()
                b.prop(self, "ht_dot_density", text="Density")
                b.prop(self, "ht_max_dot_size", text="Max Size")
                b.prop(self, "ht_min_dot_size", text="Min Size")
                b.prop(self, "ht_pattern_rotation", text="Rotation")
            col.operator("gbfx.ht_reset_settings", icon='LOOP_BACK')

    def _draw_chaos(self, layout):
        box = layout.box()
        box.label(text="4. Magical Girls Chaos", icon="COLOR")
        b = box.box()
        b.label(text="Theme Animator (Mutates actual UI Colors)", icon="COLOR")
        b.prop(self, "gsh_chaos_enabled", toggle=True, text="Enable Theme Animator")
        row = b.row(align=True)
        row.prop(self, "gsh_chaos_speed")
        row.prop(self, "gsh_chaos_base_color")
        if self.gsh_chaos_enabled:
            b.label(text=f"Animating: {len(gsh_state.chaos_targets)} matched properties", icon="INFO")
            b.operator("gbfx.gsh_rescan_theme")

        b = box.box()
        b.label(text="GPU Silhouette Outlines (Custom Draw Pass)", icon="MOD_OUTLINE")
        col = b.column(align=True)
        col.prop(self, "gsh_viewport_outline_enabled", toggle=True, text="Viewport Silhouette Outline")
        col.prop(self, "gsh_node_outline_enabled", toggle=True, text="Node Editor Outline")
        row = b.row()
        row.prop(self, "gsh_show_advanced", icon="TRIA_DOWN" if self.gsh_show_advanced else "TRIA_RIGHT", emboss=False, text="Outline Settings")
        if self.gsh_show_advanced:
            col = b.column(align=True)
            row = col.row(align=True)
            row.prop(self, "gsh_viewport_outline_width")
            row.prop(self, "gsh_viewport_outline_speed")
            row = col.row(align=True)
            row.prop(self, "gsh_viewport_outline_resolution")
            col.prop(self, "gsh_viewport_outline_depth_test")
            col.prop(self, "gsh_viewport_outline_edit_mode")
            
        b = box.box()
        b.prop(self, "gsh_border_enabled", toggle=True, text="Render/Bake Border")
        if self.gsh_border_enabled:
            row = b.row(align=True)
            row.prop(self, "gsh_border_speed")
            row.prop(self, "gsh_border_line_width")

        box.prop(self, "gbfx_debug_logging")

    def draw(self, context):
        layout = self.layout
        self._draw_palette(layout)
        self._draw_magia_baiser(layout)
        self._draw_text_outline(layout)
        self._draw_halftone(layout)
        self._draw_chaos(layout)

class GBFX_OT_clear_errors(Operator):
    bl_idname = "gbfx.clear_errors"
    bl_label = "Clear Error Log"
    def execute(self, context):
        gbfx_state.errors.clear()
        gbfx_tag_redraw({'VIEW_3D', 'UI'})
        return {"FINISHED"}

# =============================================================================
# 1. MAGIA BAISER FX
# =============================================================================

def _mb_get_color(prefs, color_key):
    if color_key == 'C1': return tuple(prefs.gbfx_color_1)
    if color_key == 'C2': return tuple(prefs.gbfx_color_2)
    if color_key == 'C3': return tuple(prefs.gbfx_color_3)
    if color_key == 'C4': return tuple(prefs.gbfx_color_4)
    if color_key == 'DARK': return tuple(prefs.gbfx_color_dark)
    return (1, 1, 1, 1)

def mb_shader():
    if mb_state.shader is None:
        mb_state.shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    return mb_state.shader

@contextmanager
def mb_alpha_blend():
    gpu.state.blend_set('ALPHA')
    try: yield
    finally: gpu.state.blend_set('NONE')

def mb_gen_star(spikes=4, inner=0.2, outer=1.0, rotation=0.0):
    k = (spikes, round(inner, 3), round(outer, 3), round(rotation, 3))
    if k in mb_state.star_cache: return mb_state.star_cache[k]
    verts = []
    for i in range(spikes):
        oa = rotation + (2 * math.pi / spikes) * i
        verts.append((math.cos(oa) * outer, math.sin(oa) * outer))
        ia = oa + math.pi / spikes
        verts.append((math.cos(ia) * inner, math.sin(ia) * inner))
    mb_state.star_cache[k] = verts
    return verts

def mb_gen_flower(petals=5, inner=0.55, outer=1.0, rotation=0.0, segments=48):
    k = (petals, round(inner, 3), round(outer, 3), round(rotation, 3), segments)
    if k in mb_state.flower_cache: return mb_state.flower_cache[k]
    verts = []
    for i in range(segments):
        th = rotation + 2 * math.pi * i / segments
        r = inner + (outer - inner) * 0.5 * (1.0 + math.cos(petals * (th - rotation)))
        verts.append((math.cos(th) * r, math.sin(th) * r))
    mb_state.flower_cache[k] = verts
    return verts

def mb_cubic_bezier(p0, p1, p2, p3, segments=16):
    pts = []
    for i in range(segments + 1):
        t = i / segments; mt = 1.0 - t
        x = (mt**3)*p0[0] + 3*(mt**2)*t*p1[0] + 3*mt*(t**2)*p2[0] + (t**3)*p3[0]
        y = (mt**3)*p0[1] + 3*(mt**2)*t*p1[1] + 3*mt*(t**2)*p2[1] + (t**3)*p3[1]
        z = (mt**3)*p0[2] + 3*(mt**2)*t*p1[2] + 3*mt*(t**2)*p2[2] + (t**3)*p3[2]
        pts.append((x, y, z))
    return pts

def mb_ease_out_cubic(t): return 1 - (1 - max(0.0, min(1.0, t))) ** 3
def mb_ease_out_back(t, s=1.70158): t = max(0.0, min(1.0, t)) - 1; return t * t * ((s + 1) * t + s) + 1

def mb_draw_polygon(verts, color, mode='TRI_FAN'):
    if len(verts) < 2: return
    shader = mb_shader()
    batch = batch_for_shader(shader, mode, {"pos": verts})
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)

def mb_draw_lines(verts, color, width=2.0, loop=False):
    gpu.state.line_width_set(width)
    mb_draw_polygon(verts, color, 'LINE_LOOP' if loop else 'LINE_STRIP')
    gpu.state.line_width_set(1.0)

def mb_draw_polygon_fill(boundary_pts, color, center):
    if len(boundary_pts) < 3: return
    mb_draw_polygon([center] + list(boundary_pts) + [boundary_pts[0]], color, 'TRI_FAN')

def mb_draw_star_glyph(co, scale, rotation, fill_color=None, outline_color=None, spikes=4, inner=0.2, outline_width=1.5):
    if scale <= 0.0: return
    verts2d = mb_gen_star(spikes, inner=inner, outer=1.0, rotation=rotation)
    pts = [(co[0] + x * scale, co[1] + y * scale) for x, y in verts2d]
    if fill_color: mb_draw_polygon_fill(pts, fill_color, (co[0], co[1]))
    if outline_color: mb_draw_lines(pts, outline_color, width=outline_width, loop=True)

class mb_FXEffect:
    def __init__(self, duration):
        self.age = 0.0; self.duration = max(duration, 0.001)
    @property
    def t(self): return min(1.0, self.age / self.duration)
    @property
    def finished(self): return self.age >= self.duration
    def advance(self, dt): self.age += dt
    def draw_view(self, context): pass
    def draw_pixel(self, context): pass

class mb_TaglioBaiserEffect(mb_FXEffect):
    def __init__(self, center, color_main, color_dark, style='MAGIA_BAISER'):
        super().__init__(duration=0.9)
        self.center = Vector(center); self.color_main = color_main
        self.color_dark = color_dark; self.style = style
        rng = random.Random(hash(tuple(round(c, 3) for c in center)) & 0xFFFF)
        self.slashes = [(0.0, rng.uniform(0.5, 1.0)), (0.12, rng.uniform(-0.5, 0.0)), (0.24, rng.uniform(1.2, 1.8))]
        self.sparks = [(rng.uniform(0, 2*math.pi), rng.uniform(0.5, 1.0), rng.uniform(0, 2*math.pi)) for _ in range(10)]

    def draw_pixel(self, context):
        co = world_to_region(context, self.center)
        if co is None: return
        with mb_alpha_blend():
            for delay, angle in self.slashes:
                local_t = (self.age - delay)
                if 0 < local_t < 0.25:
                    phase = local_t / 0.25
                    length = 45.0 * mb_ease_out_cubic(min(1.0, phase * 2.5))
                    width = 4.0 * (1.0 - phase); alpha = 1.0 - phase
                    cx, cy = co.x, co.y; ca, sa = math.cos(angle), math.sin(angle)
                    p1 = (cx + ca * length, cy + sa * length); p2 = (cx - ca * length, cy - sa * length)
                    wx, wy = -sa * width, ca * width
                    verts = [p1, (cx + wx, cy + wy), p2, (cx - wx, cy - wy)]
                    mb_draw_polygon_fill(verts, (*self.color_dark[:3], alpha * 0.95), (cx, cy))
                    mb_draw_lines(verts, (*self.color_main[:3], alpha), width=1.5, loop=True)
            burst_delay = 0.45 
            if self.age > burst_delay:
                burst_t = min(1.0, (self.age - burst_delay) / (self.duration - burst_delay))
                pop = mb_ease_out_back(min(1.0, burst_t * 2.5))
                fade = max(0.0, 1.0 - burst_t); scale = 20.0 * max(0.0, pop)
                if self.style == 'TRES_MAGIA':
                    pts = [(co.x + x * scale, co.y + y * scale) for x, y in mb_gen_flower(5, inner=0.55, outer=1.0, rotation=burst_t * 2.0)]
                    mb_draw_polygon_fill(pts, (*self.color_dark[:3], 0.85 * fade), (co.x, co.y))
                    mb_draw_lines(pts, (*self.color_main[:3], fade), width=1.5, loop=True)
                else:
                    mb_draw_star_glyph((co.x, co.y), scale, burst_t * 2.0, fill_color=(*self.color_dark[:3], 0.85 * fade), outline_color=(*self.color_main[:3], fade), inner=0.15)
                for angle, dist_f, rot in self.sparks:
                    travel = mb_ease_out_cubic(burst_t) * dist_f * 55.0
                    tx, ty = co.x + math.cos(angle) * travel, co.y + math.sin(angle) * travel
                    s = 4.0 * max(0.0, 1.0 - burst_t)
                    if self.style == 'TRES_MAGIA':
                        pts = [(tx + x * s, ty + y * s) for x, y in mb_gen_flower(5, inner=0.6, outer=1.0, rotation=rot + self.age * 5.0)]
                        mb_draw_polygon_fill(pts, (*self.color_main[:3], fade), (tx, ty))
                    else:
                        mb_draw_star_glyph((tx, ty), s, rot + self.age * 5.0, fill_color=(*self.color_main[:3], fade), inner=0.2)

class mb_EnormitaGraspEffect(mb_FXEffect):
    def __init__(self, center, dims, color_main, color_dark, intensity=1.0, style='MAGIA_BAISER'):
        super().__init__(duration=0.9)
        self.center = Vector(center); self.color_main = color_main; self.color_dark = color_dark; self.style = style
        size = max(dims.length * 0.5, 0.35)
        rng = random.Random(hash(tuple(round(c, 3) for c in center)) & 0xFFFF)
        n_vines = max(3, int(6 * intensity))
        self.vine_starts = [self.center + Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-0.4, 1))) * size for _ in range(n_vines)]
        self.jitter_seeds = [rng.random() * 10 for _ in range(n_vines)]
        self.shards = [Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))).normalized() for _ in range(max(6, int(10 * intensity)))]

    def draw_view(self, context):
        phase = min(1.0, self.t / 0.65)
        if phase >= 1.0: return
        grip = mb_ease_out_cubic(phase); fade = 1.0 - phase * 0.3
        for start, seed in zip(self.vine_starts, self.jitter_seeds):
            jitter = Vector((math.sin(seed + self.age * 9.0), math.cos(seed * 1.3 + self.age * 7.0), math.sin(seed * 0.7 + self.age * 5.0))) * (1.0 - grip) * (start - self.center).length * 0.35
            pts = mb_cubic_bezier(start, start.lerp(self.center, 0.5) + jitter, start.lerp(self.center, 0.5) + jitter, start.lerp(self.center, grip), segments=10)
            mb_draw_lines(pts, (*self.color_main[:3], self.color_main[3] * fade), width=2.0)

    def draw_pixel(self, context):
        co = world_to_region(context, self.center)
        if co is None: return
        with mb_alpha_blend():
            if self.t < 0.7:
                scale = 26.0 * max(0.0, mb_ease_out_back(min(1.0, self.t / 0.35)))
                alpha = 1.0 if self.t < 0.55 else max(0.0, 1.0 - (self.t - 0.55) / 0.15)
                verts2d = mb_gen_flower(5, inner=0.55, outer=1.0, rotation=self.age * 1.2) if self.style == 'TRES_MAGIA' else mb_gen_star(4, inner=0.2, outer=1.0, rotation=self.age * 2.0)
                verts = [(co.x + x * scale, co.y + y * scale) for x, y in verts2d]
                mb_draw_polygon_fill(verts, (*self.color_dark[:3], 0.85 * alpha), (co.x, co.y))
                mb_draw_lines(verts, self.color_main, width=1.5, loop=True)
            else:
                burst_t = mb_ease_out_cubic((self.t - 0.7) / 0.3)
                radius = 4.0 + burst_t * 46.0; alpha = max(0.0, 1.0 - burst_t)
                shard_scale = 4.5 * max(0.0, 1.0 - burst_t * 0.5)
                for i, d in enumerate(self.shards):
                    mb_draw_star_glyph((co.x + d.x * radius, co.y + d.y * radius), shard_scale, self.age * (4.0 + (i % 3)), fill_color=(*self.color_main[:3], alpha), inner=0.15)

class mb_AliceToyboxEffect(mb_FXEffect):
    def __init__(self, center, color, intensity=1.0, style='MAGIA_BAISER'):
        super().__init__(duration=0.75)
        self.center = Vector(center); self.color = color; self.style = style
        rng = random.Random(hash(tuple(round(c, 3) for c in center)) & 0xFFFF)
        self.motes = [(rng.uniform(0, 2 * math.pi), rng.uniform(30.0, 58.0), rng.uniform(0.0, 0.35), rng.uniform(0, 2 * math.pi)) for _ in range(max(5, int(9 * intensity)))]
        self.flash_rotation = rng.uniform(0, math.pi)

    def draw_pixel(self, context):
        co = world_to_region(context, self.center)
        if co is None: return
        with mb_alpha_blend():
            converge_t = mb_ease_out_cubic(min(1.0, self.t / 0.45))
            mote_alpha = max(0.0, 1.0 - max(0.0, (self.t - 0.35) / 0.15))
            if mote_alpha > 0.0:
                for angle, dist, delay, rot in self.motes:
                    local_t = max(0.0, min(1.0, (converge_t - delay) / max(1e-3, 1.0 - delay)))
                    r = dist * (1.0 - local_t)
                    mx, my = co.x + math.cos(angle) * r, co.y + math.sin(angle) * r
                    s = 3.0 + local_t * 1.5; spin = rot + self.age * 5.0
                    if self.style == 'TRES_MAGIA':
                        pts = [(mx + x * s, my + y * s) for x, y in mb_gen_flower(5, inner=0.5, outer=1.0, rotation=spin)]
                        mb_draw_polygon_fill(pts, (*self.color[:3], 0.85 * mote_alpha), (mx, my))
                    else:
                        mb_draw_star_glyph((mx, my), s, spin, fill_color=(*self.color[:3], 0.85 * mote_alpha), inner=0.15)
            flash_t = max(0.0, min(1.0, (self.t - 0.3) / 0.3))
            if flash_t > 0.0:
                pop = mb_ease_out_back(flash_t)
                scale = 30.0 * max(0.0, pop); fade = max(0.0, 1.0 - max(0.0, (self.t - 0.65) / 0.35))
                rot = self.flash_rotation + self.age * 1.4
                if self.style == 'TRES_MAGIA':
                    pts = [(co.x + x * scale, co.y + y * scale) for x, y in mb_gen_flower(5, inner=0.55, outer=1.0, rotation=rot)]
                    mb_draw_polygon_fill(pts, (*self.color[:3], 0.85 * fade), (co.x, co.y))
                    mb_draw_lines(pts, (*self.color[:3], fade), width=1.5, loop=True)
                else:
                    hot_core = (min(1.0, self.color[0]*1.25), min(1.0, self.color[1]*1.25), min(1.0, self.color[2]*1.25), 0.85*fade)
                    mb_draw_star_glyph((co.x, co.y), scale, rot, fill_color=hot_core, outline_color=(*self.color[:3], fade), inner=0.18)
                ray_alpha = max(0.0, 1.0 - flash_t) * fade
                if ray_alpha > 0.0:
                    ray_len = 12.0 + pop * 30.0
                    for i in range(8):
                        a = rot * 0.5 + i * math.pi / 4
                        mb_draw_lines([(co.x, co.y), (co.x + math.cos(a) * ray_len, co.y + math.sin(a) * ray_len)], (*self.color[:3], ray_alpha), width=1.5)

class mb_AzureRippleEffect(mb_FXEffect):
    def __init__(self, center, color):
        super().__init__(duration=0.4)
        self.center = Vector(center); self.color = color; self.rotation = random.uniform(0, math.pi / 2)
    def draw_pixel(self, context):
        co = world_to_region(context, self.center)
        if co is None: return
        pop = mb_ease_out_back(min(1.0, self.t / 0.45)); alpha = max(0.0, 1.0 - self.t); scale = 11.0 + pop * 11.0
        with mb_alpha_blend():
            mb_draw_star_glyph((co.x, co.y), scale, self.rotation, outline_color=(*self.color[:3], alpha), inner=0.12, outline_width=1.5)
            flare = scale * 1.7
            mb_draw_lines([(co.x - flare, co.y), (co.x + flare, co.y)], (*self.color[:3], alpha * 0.55), width=1.0)
            mb_draw_lines([(co.x, co.y - flare), (co.x, co.y + flare)], (*self.color[:3], alpha * 0.55), width=1.0)

class mb_ClockworkEffect(mb_FXEffect):
    def __init__(self, screen_pos, color, forward=False, intensity=1.0, style='MAGIA_BAISER'):
        super().__init__(duration=0.7)
        self.pos = screen_pos; self.color = color; self.forward = forward; self.style = style
        rng = random.Random()
        self.sparks = [(rng.uniform(0, 2 * math.pi), rng.uniform(0.5, 1.0), rng.uniform(0, 2 * math.pi)) for _ in range(max(3, int(6 * intensity)))]
    def draw_pixel(self, context):
        if context.region is None: return
        cx, cy = context.region.width * 0.5, 80.0
        direction = 1.0 if self.forward else -1.0
        pop = mb_ease_out_back(min(1.0, self.t / 0.25)); fade = max(0.0, 1.0 - max(0.0, (self.t - 0.7) / 0.3))
        r = 24.0 * pop
        with mb_alpha_blend():
            if r > 0.5:
                mb_draw_lines([(cx + math.cos(2*math.pi*i/32)*r, cy + math.sin(2*math.pi*i/32)*r) for i in range(32)], (*self.color[:3], fade * 0.6), width=1.5, loop=True)
                mb_draw_lines([(cx + math.cos(2*math.pi*i/16)*r*0.3, cy + math.sin(2*math.pi*i/16)*r*0.3) for i in range(16)], (*self.color[:3], fade * 0.3), width=1.0, loop=True)
                ha = math.pi / 2 - direction * (mb_ease_out_cubic(self.t) * math.pi * 4.0)
                mb_draw_lines([(cx, cy), (cx + math.cos(ha) * (r * 0.85), cy + math.sin(ha) * (r * 0.85))], (*self.color[:3], fade * 0.9), width=2.5)
                mb_draw_star_glyph((cx, cy), 3.0 * pop, 0, fill_color=(*self.color[:3], fade), inner=0.5)
            spin = mb_ease_out_cubic(self.t)
            for angle, dist, rot in self.sparks:
                travel = spin * dist * 45.0
                sx, sy = cx + math.cos(angle + direction * spin * 2.0) * travel, cy + math.sin(angle + direction * spin * 2.0) * travel
                s = 3.0 * max(0.0, 1.0 - spin * 0.5) * fade
                if self.style == 'TRES_MAGIA':
                    pts = [(sx + x * s, sy + y * s) for x, y in mb_gen_flower(5, inner=0.6, outer=1.0, rotation=rot + direction * self.age * 6.0)]
                    mb_draw_polygon_fill(pts, (*self.color[:3], fade), (sx, sy))
                else: mb_draw_star_glyph((sx, sy), s, rot + direction * self.age * 6.0, fill_color=(*self.color[:3], fade), inner=0.15)

class mb_BaiserFlareEffect(mb_FXEffect):
    def __init__(self, screen_pos, color_main, color_crack):
        super().__init__(duration=0.5)
        self.pos = screen_pos; self.color_main = color_main; self.color_crack = color_crack
        rng = random.Random()
        self.embers = [(rng.uniform(0, 2 * math.pi), rng.uniform(0.5, 1.0), rng.uniform(0, 2 * math.pi)) for _ in range(5)]
    def draw_pixel(self, context):
        if context.region is None: return
        cx, cy = self.pos if self.pos is not None else (context.region.width * 0.5, 130.0)
        alpha = max(0.0, 1.0 - self.t / 0.5); r = 10.0 + mb_ease_out_back(min(1.0, self.t / 0.25)) * 28.0
        with mb_alpha_blend():
            mb_draw_star_glyph((cx, cy), r, self.age * 3.0, fill_color=(*self.color_main[:3], 0.35 * alpha), outline_color=(*self.color_main[:3], alpha), spikes=4, inner=0.28, outline_width=2.0)
            ember_t = mb_ease_out_cubic(min(1.0, self.t / 0.4))
            for angle, dist, rot in self.embers:
                travel = ember_t * dist * 30.0
                ex, ey = cx + math.cos(angle) * travel, cy + math.sin(angle) * travel - travel * 0.15
                mb_draw_star_glyph((ex, ey), 3.0 * max(0.0, 1.0 - ember_t * 0.5), rot + self.age * 5.0, fill_color=(*self.color_main[:3], alpha), spikes=4, inner=0.15)
            w, h, inset = context.region.width, context.region.height, 3.0
            mb_draw_lines([(inset, inset), (w - inset, inset), (w - inset, h - inset), (inset, h - inset)], (*self.color_crack[:3], alpha * 0.5), width=1.5, loop=True)

class mb_EnormitaSaluteEffect(mb_FXEffect):
    def __init__(self, color, style='MAGIA_BAISER'):
        super().__init__(duration=1.2)
        self.color = color; self.style = style; rng = random.Random()
        self.twinkles = [(rng.uniform(0, 2 * math.pi), rng.uniform(1.8, 2.8), rng.uniform(0.0, 0.25)) for _ in range(4)]
    def draw_pixel(self, context):
        if context.region is None: return
        cx, cy = context.region.width - 50.0, 50.0
        pop = mb_ease_out_back(min(1.0, self.t / 0.35)); alpha = 1.0 if self.t < 0.75 else max(0.0, 1.0 - (self.t - 0.75) / 0.25)
        scale = 16.0 * max(0.0, pop)
        with mb_alpha_blend():
            if scale > 0:
                ripple_t = mb_ease_out_cubic(min(1.0, self.t / 0.4))
                ripple_alpha = max(0.0, 1.0 - ripple_t) * alpha
                if ripple_alpha > 0.0:
                    ripple_r = 15.0 + ripple_t * 35.0
                    mb_draw_lines([(cx + math.cos(2*math.pi*i/32)*ripple_r, cy + math.sin(2*math.pi*i/32)*ripple_r) for i in range(32)], (*self.color[:3], ripple_alpha), width=1.5, loop=True)
                arc_r = scale * 1.5; spin_angle = mb_ease_out_cubic(self.t) * math.pi * 1.5
                for i in range(4):
                    a1 = spin_angle + i * (math.pi / 2); a2 = a1 + (math.pi / 3)
                    mb_draw_lines([(cx + math.cos(a) * arc_r, cy + math.sin(a) * arc_r) for a in [a1 + (a2-a1)*j/8 for j in range(9)]], (*self.color[:3], alpha * 0.8), width=1.5)
                    mb_draw_star_glyph((cx + math.cos(a1)*arc_r, cy + math.sin(a1)*arc_r), 3.0 * alpha, a1, fill_color=(*self.color[:3], alpha), spikes=4, inner=0.2)
                mb_draw_star_glyph((cx, cy), scale, 0.3 + spin_angle*0.2, fill_color=(*self.color[:3], 0.9 * alpha), outline_color=(*self.color[:3], alpha), spikes=4, inner=0.2, outline_width=1.0)
                for angle, dist_f, delay in self.twinkles:
                    tw_t = max(0.0, min(1.0, (self.t - delay) / 0.4))
                    if (tw_alpha := (1.0 - tw_t) * alpha) > 0.0:
                        mb_draw_star_glyph((cx + math.cos(angle + spin_angle*0.5) * (scale * dist_f), cy + math.sin(angle + spin_angle*0.5) * (scale * dist_f)), max(0.0, 4.0 * mb_ease_out_back(tw_t)), angle + self.age * 4.0, fill_color=(*self.color[:3], 0.85 * tw_alpha), spikes=4, inner=0.1)
            blf.size(0, 14)
            blf.color(0, self.color[0], self.color[1], self.color[2], alpha)
            blf.position(0, cx - 85, cy - 5, 0)
            blf.draw(0, "Saved \u22c6")

# --- CENTRALIZED TRIGGER DISPATCHER ---
def _mb_dispatch_effect(context, effect_factory, trigger_flag, force=False):
    prefs = get_prefs(context)
    if prefs is None or not (force or (prefs.mb_enable_fx and getattr(prefs, trigger_flag, False))):
        return
        
    # LIMIT MAX EFFECTS to 4 to ensure smooth 60fps even during mass destruction
    if len(mb_state.effects) >= 4:
        mb_state.effects.pop(0)
        
    mb_state.effects.append(effect_factory(prefs, context))

def mb_trigger_delete(context, center, dims, force=False):
    _mb_dispatch_effect(context, lambda p, c: mb_TaglioBaiserEffect(center, _mb_get_color(p, 'C1'), _mb_get_color(p, 'DARK'), style=p.mb_style_preset), 'mb_trig_delete', force)
def mb_trigger_add(context, center, force=False):
    _mb_dispatch_effect(context, lambda p, c: mb_AliceToyboxEffect(center, _mb_get_color(p, 'C4'), p.mb_intensity, style=p.mb_style_preset), 'mb_trig_add', force)
def mb_trigger_select(context, center, force=False):
    _mb_dispatch_effect(context, lambda p, c: mb_AzureRippleEffect(center, _mb_get_color(p, 'C3')), 'mb_trig_select', force)
def mb_trigger_apply_modifier(context, center, dims, force=False):
    _mb_dispatch_effect(context, lambda p, c: mb_EnormitaGraspEffect(center, dims, _mb_get_color(p, 'C2'), _mb_get_color(p, 'DARK'), p.mb_intensity, style=p.mb_style_preset), 'mb_trig_apply_modifier', force)
def mb_trigger_undo(context, screen_pos=None, force=False):
    _mb_dispatch_effect(context, lambda p, c: mb_ClockworkEffect(screen_pos, _mb_get_color(p, 'C4'), forward=False, intensity=p.mb_intensity, style=p.mb_style_preset), 'mb_trig_undo', force)
def mb_trigger_redo(context, screen_pos=None, force=False):
    _mb_dispatch_effect(context, lambda p, c: mb_ClockworkEffect(screen_pos, _mb_get_color(p, 'C2'), forward=True, intensity=p.mb_intensity, style=p.mb_style_preset), 'mb_trig_redo', force)
def mb_trigger_error(context, screen_pos=None, force=False):
    _mb_dispatch_effect(context, lambda p, c: mb_BaiserFlareEffect(screen_pos, _mb_get_color(p, 'C1'), _mb_get_color(p, 'C1')), 'mb_trig_error', force)
def mb_trigger_save(context, force=False):
    _mb_dispatch_effect(context, lambda p, c: mb_EnormitaSaluteEffect(_mb_get_color(p, 'C2'), style=p.mb_style_preset), 'mb_trig_save', force)

def mb_draw_callback_view():
    context = bpy.context
    if context.area and getattr(context.area, "type", "") == 'VIEW_3D' and context.region:
        for eff in mb_state.effects: eff.draw_view(context)

def mb_draw_callback_pixel():
    context = bpy.context
    if context.area and getattr(context.area, "type", "") == 'VIEW_3D' and context.region:
        for eff in mb_state.effects: eff.draw_pixel(context)

def mb_fx_tick():
    try:
        if not mb_state.running: return None
        if mb_state.effects:
            for eff in mb_state.effects: eff.advance(0.016)
            mb_state.effects = [e for e in mb_state.effects if not e.finished]
            gbfx_tag_redraw({'VIEW_3D'})
            return 0.016
        return 0.1
    except Exception as e:
        gbfx_state.log_error("mb_fx_tick", e)
        return 0.1

def mb__rebuild_object_cache(scene, depsgraph=None):
    cache = {}
    for ob in scene.objects:
        center, size = bbox_center_and_size(ob, depsgraph)
        cache[ob.name] = {'pos': center, 'dim': size}
    mb_state.obj_cache = cache

@persistent
def mb_on_depsgraph_update(scene, depsgraph):
    if not mb_state.running: return
    context = bpy.context
    prefs = get_prefs(context)
    if not prefs or not prefs.mb_enable_fx: return
    
    current_count = len(scene.objects)
    
    # Gate caching loops behind relevant triggers to save CPU
    if prefs.mb_trig_delete or prefs.mb_trig_add:
        if current_count != mb_state.obj_count:
            mb_state.obj_count = current_count
            if time.time() >= mb_state.suppress_until:
                current_names = {ob.name for ob in scene.objects}
                old_cache = mb_state.obj_cache
                deleted = set(old_cache.keys()) - current_names
                added = current_names - set(old_cache.keys())
                
                # Bulk Operations Cheat: Averages positions to spawn 1 effect instead of 1000
                if deleted:
                    if len(deleted) > 5:
                        avg_pos = sum((old_cache[n]['pos'] for n in deleted), Vector()) / len(deleted)
                        mb_trigger_delete(context, avg_pos, Vector((1,1,1)))
                    else:
                        for name in deleted:
                            info = old_cache[name]
                            mb_trigger_delete(context, info['pos'], info['dim'])
                            
                if added:
                    if len(added) > 5:
                        valid_objs = [scene.objects.get(n) for n in added if scene.objects.get(n)]
                        if valid_objs:
                            avg_pos = sum((bbox_center_and_size(ob, depsgraph)[0] for ob in valid_objs), Vector()) / len(valid_objs)
                            mb_trigger_add(context, avg_pos)
                    else:
                        for name in added:
                            ob = scene.objects.get(name)
                            if ob is not None:
                                center, _size = bbox_center_and_size(ob, depsgraph)
                                mb_trigger_add(context, center)
                                
            mb__rebuild_object_cache(scene, depsgraph)
        else:
            cache = mb_state.obj_cache
            for update in depsgraph.updates:
                if isinstance(update.id, bpy.types.Object) and (update.is_updated_transform or update.is_updated_geometry):
                    ob = scene.objects.get(update.id.name)
                    if ob is not None:
                        center, size = bbox_center_and_size(ob, depsgraph)
                        cache[ob.name] = {'pos': center, 'dim': size}
                        
    # BEST-EFFORT WORKAROUND: In Blender, there is no direct event for "modifier applied".
    # We inspect window_manager.operators history as a heuristic proxy.
    wm = getattr(context, "window_manager", None)
    ops = getattr(wm, "operators", []) if wm else []
    if ops:
        last_op = ops[-1]
        op_id = (last_op.bl_idname, len(ops))
        if last_op.bl_idname == 'OBJECT_OT_modifier_apply' and op_id != mb_state.last_modifier_op_id:
            mb_state.last_modifier_op_id = op_id
            ob = context.active_object
            if ob is not None:
                applied_name = getattr(last_op, 'modifier', '')
                really_applied = not applied_name or ob.modifiers.get(applied_name) is None
                if really_applied:
                    center, size = bbox_center_and_size(ob, depsgraph)
                    mb_trigger_apply_modifier(context, center, size)

@persistent
def mb_on_undo_post(*args, **kwargs):
    if not mb_state.running: return
    mb_state.suppress_until = time.time() + 0.2
    context = bpy.context
    mb_trigger_undo(context)
    if getattr(context, "scene", None):
        mb_state.obj_count = len(context.scene.objects)
        mb__rebuild_object_cache(context.scene, _current_depsgraph())

@persistent
def mb_on_redo_post(*args, **kwargs):
    if not mb_state.running: return
    mb_state.suppress_until = time.time() + 0.2
    context = bpy.context
    mb_trigger_redo(context)
    if getattr(context, "scene", None):
        mb_state.obj_count = len(context.scene.objects)
        mb__rebuild_object_cache(context.scene, _current_depsgraph())

@persistent
def mb_on_save_post(*args, **kwargs):
    if not mb_state.running: return
    mb_trigger_save(bpy.context)

@persistent
def mb_on_load_post(*args, **kwargs):
    mb_state.obj_cache = {}
    mb_state.obj_count = 0
    mb_state.effects.clear()
    mb_state.last_modifier_op_id = None
    mb_state.last_selection_hash = None
    if mb_state.running:
        if getattr(bpy.context, "scene", None):
            mb__rebuild_object_cache(bpy.context.scene, _current_depsgraph())
            mb_state.obj_count = len(bpy.context.scene.objects)
        mb__subscribe_msgbus()

def mb__on_active_object_changed():
    context = bpy.context
    prefs = get_prefs(context)
    if prefs is None or not (prefs.mb_enable_fx and prefs.mb_trig_select and prefs.mb_selection_tracking_mode == 'ACTIVE_ONLY'): return
    ob = context.view_layer.objects.active
    if ob is not None and ob.visible_get():
        center, _size = bbox_center_and_size(ob, _current_depsgraph())
        mb_trigger_select(context, center)

def mb__subscribe_msgbus():
    try: bpy.msgbus.clear_by_owner(mb_state.msgbus_owner)
    except Exception: pass
    try:
        bpy.msgbus.subscribe_rna(
            key=(bpy.types.LayerObjects, "active"), owner=mb_state.msgbus_owner, args=(), notify=mb__on_active_object_changed,
        )
    except Exception as e: gbfx_state.log_error("mb_subscribe_msgbus", e)

def mb_selection_poll_tick():
    if not mb_state.running: return None
    context = bpy.context
    prefs = get_prefs(context)
    if prefs is None or not (prefs.mb_enable_fx and prefs.mb_trig_select and prefs.mb_selection_tracking_mode == 'FULL_SELECTION'): return 0.15
    try:
        sel_hash = frozenset(ob.name for context in [context] for ob in context.selected_objects)
        if sel_hash != mb_state.last_selection_hash:
            mb_state.last_selection_hash = sel_hash
            ob = context.active_object or (context.selected_objects[0] if context.selected_objects else None)
            if ob is not None and ob.visible_get():
                center, _size = bbox_center_and_size(ob, _current_depsgraph())
                mb_trigger_select(context, center)
    except Exception as e: gbfx_state.log_error("mb_selection_poll_tick", e)
    return 0.15

def mb__delayed_startup():
    scene = getattr(bpy.context, "scene", None)
    if scene is not None:
        mb__rebuild_object_cache(scene)
        mb_state.obj_count = len(scene.objects)
    return None

class MBFX_OT_apply_all_modifiers(Operator):
    bl_idname = "gbfx.mb_apply_all_modifiers"
    bl_label = "Apply All Modifiers (FX)"
    bl_options = {'REGISTER', 'UNDO'}
    @classmethod
    def poll(cls, context):
        return context.active_object is not None and len(context.active_object.modifiers) > 0
    def execute(self, context):
        ob = context.active_object
        any_failed = False
        for mod in list(ob.modifiers):
            try:
                with context.temp_override(object=ob):
                    result = bpy.ops.object.modifier_apply(modifier=mod.name)
            except RuntimeError:
                result = {'CANCELLED'}
            if 'CANCELLED' in result:
                any_failed = True
        if any_failed:
            mb_trigger_error(context)
            self.report({'WARNING'}, "Some modifiers could not be applied")
        else:
            self.report({'INFO'}, "All Modifiers Applied")
        return {'FINISHED'}

class MBFX_OT_preview_effect(Operator):
    bl_idname = "gbfx.mb_preview_effect"
    bl_label = "Preview Effect"
    bl_description = "Play the selected effect once, without doing the real action"
    bl_options = {'INTERNAL'}
    @classmethod
    def poll(cls, context): return get_prefs(context) is not None
    def execute(self, context):
        prefs = get_prefs(context)
        effect = prefs.mb_preview_effect
        ob = context.active_object
        if ob is not None:
            center, dims = bbox_center_and_size(ob, _current_depsgraph())
        else:
            center, dims = context.scene.cursor.location.copy(), Vector((0.4, 0.4, 0.4))
        screen_pos = world_to_region(context, center)
        if effect == 'DELETE': mb_trigger_delete(context, center, dims, force=True)
        elif effect == 'ADD': mb_trigger_add(context, center, force=True)
        elif effect == 'SELECT': mb_trigger_select(context, center, force=True)
        elif effect == 'APPLY_MODIFIER': mb_trigger_apply_modifier(context, center, dims, force=True)
        elif effect == 'UNDO': mb_trigger_undo(context, screen_pos=screen_pos, force=True)
        elif effect == 'REDO': mb_trigger_redo(context, screen_pos=screen_pos, force=True)
        elif effect == 'ERROR': mb_trigger_error(context, screen_pos=screen_pos, force=True)
        elif effect == 'SAVE': mb_trigger_save(context, force=True)
        return {'FINISHED'}


# =============================================================================
# 2. UI TEXT OUTLINE MODULE
# =============================================================================

uto_STYLE_CATEGORIES = ("widget", "widget_label", "panel_title", "tooltip")
uto_SHADOW_MODE = {"SOFT": 3, "WIDE": 5, "CRISP": 6}

def uto_resolve_shadow_mode(prefs):
    return uto_SHADOW_MODE.get(prefs.uto_outline_style, 6)

def uto_resolve_shadow_value(prefs, context=None):
    return 0.0 if prefs.uto_color_mode == "DARK" else 1.0

def uto_apply_outline(context=None):
    context = context or bpy.context
    prefs = get_prefs(context)
    if prefs is None: return
    try: style = context.preferences.ui_styles[0]
    except (IndexError, AttributeError): return

    if not prefs.uto_enabled:
        uto_disable_outline(context)
        return

    shadow_mode = uto_resolve_shadow_mode(prefs)
    shadow_value = uto_resolve_shadow_value(prefs, context)
    shadow_alpha = prefs.uto_intensity

    for category in uto_STYLE_CATEGORIES:
        if not getattr(prefs, "uto_apply_%s" % category, True): continue
        font_style = getattr(style, category, None)
        if font_style is None: continue
        uto__write_font_style(font_style, shadow_mode, shadow_value, shadow_alpha)

    gbfx_tag_redraw({'VIEW_3D', 'NODE_EDITOR', 'IMAGE_EDITOR', 'PROPERTIES', 'OUTLINER', 'UI'})

def uto_disable_outline(context=None):
    context = context or bpy.context
    try: style = context.preferences.ui_styles[0]
    except (IndexError, AttributeError): return

    for category in uto_STYLE_CATEGORIES:
        font_style = getattr(style, category, None)
        if font_style is None: continue
        uto__write_font_style(font_style, 0, 0.0, 0.0)

    gbfx_tag_redraw({'VIEW_3D', 'NODE_EDITOR', 'IMAGE_EDITOR', 'PROPERTIES', 'OUTLINER', 'UI'})

def uto__write_font_style(font_style, shadow_mode, shadow_value, shadow_alpha):
    try: font_style.shadow = shadow_mode
    except Exception:
        try: font_style.shadow = 5
        except Exception: pass
    font_style.shadow_alpha = shadow_alpha
    font_style.shadow_value = shadow_value
    font_style.shadow_offset_x = 0
    font_style.shadow_offset_y = 0

class UITEXTOUTLINE_OT_refresh(Operator):
    bl_idname = "gbfx.uto_refresh"
    bl_label = "Refresh UI Text Outline"
    bl_options = {"REGISTER"}
    def execute(self, context):
        uto_apply_outline(context)
        self.report({"INFO"}, "UI Text Outline Refreshed")
        return {"FINISHED"}

@persistent
def uto__on_load_post(_dummy):
    uto_apply_outline()


# =============================================================================
# 3. HALFTONE OVERLAY MODULE
# =============================================================================

ht_VERTEX_SOURCE = """
void main() {
    uvInterp = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, u_far_z, 1.0);
}
"""

ht_FRAGMENT_SOURCE = """
vec2 rotate(vec2 v, float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return vec2(v.x * c - v.y * s, v.x * s + v.y * c);
}

void main() {
    float val;
    if (u_direction < 0.5) val = 1.0 - uvInterp.y;                                   
    else if (u_direction < 1.5) val = 1.0 - uvInterp.x;                                   
    else if (u_direction < 2.5) val = 1.0 - clamp((uvInterp.x + uvInterp.y) * 0.5, 0.0, 1.0); 
    else {
        vec2 centered = uvInterp - 0.5;
        centered.x *= u_resolution.x / max(u_resolution.y, 1.0);
        val = length(centered) * 1.4;
    }

    float t = smoothstep(1.0 - u_coverage, 2.0 - u_coverage, val);
    float radius = mix(u_min_dot, u_max_dot, t);
    
    if (radius <= 0.001) discard;

    vec2 px = uvInterp * u_resolution;
    vec2 cellUV = fract(rotate(px, u_rotation) * u_density) - 0.5;

    float dist = length(cellUV);
    float aa = fwidth(dist) + 1e-5;
    float dotMask = 1.0 - smoothstep(radius - aa, radius + aa, dist);

    if (dotMask <= 0.001) discard;
    fragColor = vec4(u_dot_color.rgb, dotMask * u_dot_color.a * u_opacity);
}
"""

def ht_create_shader():
    vert_out = gpu.types.GPUStageInterfaceInfo("halftone_bg_iface")
    vert_out.smooth('VEC2', "uvInterp")
    shader_info = gpu.types.GPUShaderCreateInfo()
    shader_info.push_constant('VEC4', "u_dot_color")
    shader_info.push_constant('VEC2', "u_resolution")
    shader_info.push_constant('FLOAT', "u_opacity")
    shader_info.push_constant('FLOAT', "u_density")
    shader_info.push_constant('FLOAT', "u_max_dot")
    shader_info.push_constant('FLOAT', "u_min_dot")
    shader_info.push_constant('FLOAT', "u_rotation")
    shader_info.push_constant('FLOAT', "u_direction")
    shader_info.push_constant('FLOAT', "u_coverage")
    shader_info.push_constant('FLOAT', "u_far_z")
    shader_info.vertex_in(0, 'VEC2', "pos")
    shader_info.vertex_out(vert_out)
    shader_info.fragment_out(0, 'VEC4', "fragColor")
    shader_info.vertex_source(ht_VERTEX_SOURCE)
    shader_info.fragment_source(ht_FRAGMENT_SOURCE)
    return gpu.shader.create_from_info(shader_info)

def ht_tag_redraw_all_view3d():
    gbfx_tag_redraw({'VIEW_3D'})

ht_DIRECTION_TO_FLOAT = {'VERTICAL': 0.0, 'HORIZONTAL': 1.0, 'DIAGONAL': 2.0, 'VIGNETTE': 3.0}

def ht_get_shader_and_batch():
    if ht_state.shader is None:
        ht_state.shader = ht_create_shader()
        coords = ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0))
        indices = ((0, 1, 2), (2, 1, 3))
        ht_state.batch = batch_for_shader(ht_state.shader, 'TRIS', {"pos": coords}, indices=indices)
    return ht_state.shader, ht_state.batch

def ht_draw_callback():
    blend_changed = False
    try:
        context = bpy.context
        region = context.region
        space = getattr(context, "space_data", None)
        if region is None or space is None: return
        
        if space.type == 'VIEW_3D' and hasattr(space, "shading"):
            shading = space.shading
            if shading.type in {'MATERIAL', 'RENDERED'}: return
            is_hdri_bg = False
            if shading.type == 'SOLID' and getattr(shading, "background_type", 'THEME') == 'WORLD': is_hdri_bg = True
            if is_hdri_bg: return
            
        settings = get_prefs()
        if settings is None or not settings.ht_enabled: return
        if region.width <= 0 or region.height <= 0: return

        shader, batch = ht_get_shader_and_batch()
        proj = gpu.matrix.get_projection_matrix()
        v_far = proj @ Vector((0.0, 0.0, -10000.0, 1.0))
        far_z = (v_far.z / v_far.w) if v_far.w != 0.0 else 0.99999
        far_z = 0.99999 if far_z > 0.5 else 0.00001 

        direction = ht_DIRECTION_TO_FLOAT.get(settings.ht_gradient_direction, 3.0)

        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('LESS_EQUAL')
        blend_changed = True

        shader.bind()
        shader.uniform_float("u_dot_color", tuple(settings.ht_dot_color))
        shader.uniform_float("u_resolution", (float(region.width), float(region.height)))
        shader.uniform_float("u_opacity", settings.ht_opacity)
        shader.uniform_float("u_density", settings.ht_dot_density)
        shader.uniform_float("u_max_dot", settings.ht_max_dot_size)
        shader.uniform_float("u_min_dot", settings.ht_min_dot_size)
        shader.uniform_float("u_rotation", settings.ht_pattern_rotation)
        shader.uniform_float("u_direction", direction)
        shader.uniform_float("u_coverage", settings.ht_mask_coverage)
        shader.uniform_float("u_far_z", far_z)

        batch.draw(shader)

    except Exception as exc:
        gbfx_state.log_error("Halftone Draw Error", traceback.format_exc())
    finally:
        if blend_changed:
            gpu.state.blend_set('NONE')
            gpu.state.depth_test_set('NONE')

class HALFTONE_OT_reset_settings(Operator):
    bl_idname = "gbfx.ht_reset_settings"
    bl_label = "Reset to Defaults"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        settings = get_prefs()
        if settings:
            for prop_id in type(settings).bl_rna.properties.keys():
                if prop_id.startswith("ht_"):
                    settings.property_unset(prop_id)
            ht_tag_redraw_all_view3d()
            self.report({'INFO'}, "Halftone Settings Reset")
        return {'FINISHED'}


# =============================================================================
# 4. GUSHING CHAOS (THEME ANIMATOR, OUTLINES, BORDER)
# =============================================================================

def gsh__build_banded_segments(polygon, phase01, n_colors, step_px=6.0, max_segments=400):
    n = len(polygon)
    if n < 2: return {}
    edges = [(polygon[i], polygon[(i + 1) % n]) for i in range(n)]
    lengths = [math.dist(a, b) for a, b in edges]
    total = sum(lengths)
    if total <= 1e-6: return {}

    approx_segs = max(n_colors * 4, int(total / step_px))
    approx_segs = min(approx_segs, max_segments)
    seg_len = total / approx_segs

    buckets = {i: [] for i in range(n_colors)}
    samples = [(0.0, polygon[0])]
    cum = 0.0
    edge_i = 0
    for s in range(1, approx_segs + 1):
        target = s * seg_len
        while edge_i < n and cum + lengths[edge_i] < target - 1e-9:
            cum += lengths[edge_i]
            edge_i += 1
        if edge_i >= n:
            samples.append((total, polygon[0]))
            continue
        a, b = edges[edge_i]
        edge_len = lengths[edge_i]
        t = 0.0 if edge_len < 1e-9 else (target - cum) / edge_len
        t = min(max(t, 0.0), 1.0)
        pt = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        samples.append((target, pt))

    for i in range(len(samples) - 1):
        u0, p0 = samples[i]
        u1, p1 = samples[i + 1]
        mid_u = ((u0 + u1) / 2.0) / total
        band = int(((mid_u + phase01) % 1.0) * n_colors) % n_colors
        buckets[band].append((p0, p1))
    return buckets

def gsh__draw_rotating_outline(polygon, phase01, colors, line_width=3.0):
    if not polygon or len(polygon) < 2: return
    buckets = gsh__build_banded_segments(polygon, phase01, len(colors))
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    
    gpu.state.blend_set("ALPHA")
    gpu.state.line_width_set(line_width)
    for band, segs in buckets.items():
        if not segs: continue
        pos = []
        for p0, p1 in segs:
            pos.append(p0)
            pos.append(p1)
        batch = batch_for_shader(shader, "LINES", {"pos": pos})
        shader.bind()
        shader.uniform_float("color", colors[band])
        batch.draw(shader)
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set("NONE")

def gsh_rescan_theme(prefs):
    gsh_state.chaos_targets.clear()
    base = prefs.gsh_chaos_base_color
    tr, tg, tb = float(base[0]), float(base[1]), float(base[2])
    
    ALLOWED_IDENTIFIERS = {"ThemeView3D", "ThemeNodeEditor", "ThemeOutliner", "ThemeUserInterface", "ThemeProperties"}

    def traverse(obj, depth=0):
        if depth > 15 or not hasattr(obj, "bl_rna"): return
        obj_type = obj.bl_rna.identifier
        for prop in obj.bl_rna.properties:
            if prop.identifier == "rna_type": continue
            try: val = getattr(obj, prop.identifier)
            except Exception as e: gbfx_state.log_error("rescan_theme getattr", e); continue
                
            if prop.type == 'POINTER' and val: traverse(val, depth + 1)
            elif prop.type == 'COLLECTION' and val:
                for item in val: traverse(item, depth + 1)
            elif prop.type == 'FLOAT' and getattr(prop, "subtype", "") in {'COLOR', 'COLOR_GAMMA'}:
                if prop.is_readonly: continue
                if prefs.gsh_viewport_outline_enabled and obj_type == "ThemeView3D" and prop.identifier in ("object_active", "editmesh_active"): continue
                if getattr(prefs, "gsh_node_outline_enabled", False) and obj_type == "ThemeNodeEditor" and prop.identifier == "node_active": continue
                size = len(val)
                if size not in (3, 4): continue
                    
                is_target = False
                if obj_type == "ThemeView3D" and prop.identifier in ("object_active", "editmesh_active"): is_target = True
                elif obj_type == "ThemeNodeEditor" and prop.identifier == "node_active": is_target = True
                elif obj_type == "ThemeOutliner" and prop.identifier == "active": is_target = True
                    
                # Constrained heuristic: only scan safe theme components and use strict distance (0.02)
                if not is_target and obj_type in ALLOWED_IDENTIFIERS:
                    v0, v1, v2 = float(val[0]), float(val[1]), float(val[2])
                    dist = math.sqrt((v0 - tr)**2 + (v1 - tg)**2 + (v2 - tb)**2)
                    if dist < 0.02: is_target = True
                        
                if is_target:
                    v0, v1, v2 = float(val[0]), float(val[1]), float(val[2])
                    alpha = float(val[3]) if size == 4 else 1.0
                    gsh_state.chaos_targets.append((obj, prop.identifier, v0, v1, v2, alpha, size))
    for theme in bpy.context.preferences.themes: traverse(theme)

def gsh_restore_theme():
    for obj, prop_id, orig_r, orig_g, orig_b, orig_a, size in gsh_state.chaos_targets:
        try:
            if size == 3: setattr(obj, prop_id, (orig_r, orig_g, orig_b))
            else: setattr(obj, prop_id, (orig_r, orig_g, orig_b, orig_a))
        except Exception as e: gbfx_state.log_error("restore_theme setattr", e)
    gsh_state.chaos_targets.clear()

def gsh_rescan_if_enabled():
    try:
        prefs = get_prefs(bpy.context)
        if prefs and prefs.gsh_chaos_enabled:
            gsh_restore_theme()
            gsh_rescan_theme(prefs)
    except Exception as e: gbfx_state.log_error("_rescan_if_enabled", e)

class GUSHING_OT_rescan_theme(Operator):
    bl_idname = "gbfx.gsh_rescan_theme"
    bl_label = "Force Rescan Theme Colors"
    bl_description = "Force the addon to rescan the theme for properties matching the Target Theme Color"
    def execute(self, context):
        prefs = get_prefs(context)
        if not prefs: return {"CANCELLED"}
        if prefs.gsh_chaos_enabled:
            gsh_restore_theme(); gsh_rescan_theme(prefs)
            self.report({"INFO"}, f"Found {len(gsh_state.chaos_targets)} theme properties.")
        else: self.report({"WARNING"}, "Enable Magical Girls Chaos first!")
        return {"FINISHED"}

def gsh__get_outline_depth_shader():
    if gsh_state.depth_shader is None:
        info = gpu.types.GPUShaderCreateInfo()
        info.push_constant('MAT4', "viewProjectionMatrix")
        info.push_constant('MAT4', "modelMatrix")
        info.vertex_in(0, 'VEC3', "pos")
        info.vertex_source("void main() { gl_Position = viewProjectionMatrix * modelMatrix * vec4(pos, 1.0); }")
        info.fragment_source("void main() {}")
        gsh_state.depth_shader = gpu.shader.create_from_info(info)
    return gsh_state.depth_shader

def gsh__get_outline_mask_shader():
    if gsh_state.mask_shader is None:
        info = gpu.types.GPUShaderCreateInfo()
        info.push_constant('MAT4', "viewProjectionMatrix")
        info.push_constant('MAT4', "modelMatrix")
        info.vertex_in(0, 'VEC3', "pos")
        info.fragment_out(0, 'VEC4', "fragColor")
        info.vertex_source("void main() { gl_Position = viewProjectionMatrix * modelMatrix * vec4(pos, 1.0); }")
        info.fragment_source("void main() { fragColor = vec4(1.0, 1.0, 1.0, 1.0); }")
        gsh_state.mask_shader = gpu.shader.create_from_info(info)
    return gsh_state.mask_shader

def gsh__get_outline_compose_shader():
    if gsh_state.compose_shader is None:
        info = gpu.types.GPUShaderCreateInfo()
        info.sampler(0, 'FLOAT_2D', "mask_tex")
        info.push_constant('VEC2', "u_resolution")
        info.push_constant('VEC2', "texel_size")
        info.push_constant('FLOAT', "outline_px")
        info.push_constant('VEC2', "pivot_screen")
        info.push_constant('FLOAT', "phase01")
        info.push_constant('VEC4', "color0")
        info.push_constant('VEC4', "color1")
        info.push_constant('VEC4', "color2")
        info.push_constant('VEC4', "color3")
        info.vertex_in(0, 'VEC2', "pos")
        vert_out = gpu.types.GPUStageInterfaceInfo("compose_iface")
        vert_out.smooth('VEC2', "v_uv")
        vert_out.smooth('VEC2', "v_screen_pos")
        info.vertex_out(vert_out)
        info.fragment_out(0, 'VEC4', "fragColor")
        info.vertex_source("""
void main() {
    v_screen_pos = pos; v_uv = pos / u_resolution; vec2 ndc = v_uv * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}""")
        info.fragment_source("""
vec4 wheel_color(float t) {
    vec4 colors[4] = vec4[4](color0, color1, color2, color3);
    float idx = fract(t) * 4.0; return colors[int(idx) % 4];
}
void main() {
    float center = texture(mask_tex, v_uv).r; float edge = 0.0;
    const int SAMPLES = 12;
    for (int i = 0; i < SAMPLES; i++) {
        float a = 6.28318530718 * float(i) / float(SAMPLES);
        vec2 offset = vec2(cos(a), sin(a)) * outline_px * texel_size;
        float s = texture(mask_tex, v_uv + offset).r;
        edge = max(edge, abs(s - center));
    }
    if (edge < 0.02) discard;
    float angle = atan(v_screen_pos.y - pivot_screen.y, v_screen_pos.x - pivot_screen.x);
    float t = angle / 6.28318530718 + phase01;
    vec4 col = wheel_color(t);
    fragColor = vec4(col.rgb, edge * col.a);
}""")
        gsh_state.compose_shader = gpu.shader.create_from_info(info)
    return gsh_state.compose_shader

@persistent
def gsh_outline_on_depsgraph_update(scene, depsgraph):
    prefs = get_prefs(bpy.context)
    if not prefs or (not prefs.gsh_viewport_outline_enabled and not prefs.gsh_node_outline_enabled):
        return

    for update in depsgraph.updates:
        if isinstance(update.id, bpy.types.Object):
            # is_updated_geometry is documented as tracking the *object's*
            # evaluated geometry, so it's the correct signal here: a pure
            # transform/selection update (e.g. dragging the Move/Rotate/Scale
            # gizmo) leaves it False, and local-space geometry is untouched --
            # matrix_world is re-applied fresh every draw as a shader uniform,
            # so there's nothing to rebuild. Treating every transform tick as
            # "needs rebuild" used to force a full to_mesh() + modifier
            # re-evaluation + fresh GPU buffer upload on *every* mouse-move
            # frame of *any* move/rotate/scale -- exactly what starves the
            # redraws that Active-Tool gizmo hover/drag interaction depends on.
            if update.is_updated_geometry:
                gsh_state.batch_dirty.add(update.id.name)
        elif isinstance(update.id, (bpy.types.Mesh, bpy.types.Curve)):
            # NOTE: is_updated_geometry is NOT a reliable signal here -- it's
            # documented as tracking object geometry, not datablock updates,
            # and is often simply unset for Mesh/Curve IDs (e.g. edits made
            # while in Edit Mode, or right after leaving it, were being missed
            # entirely when this branch was gated by that flag, which left the
            # outline stuck showing the mesh's shape from *before* the edit).
            # A Mesh/Curve ID showing up in depsgraph.updates at all already
            # means something about it changed, so invalidate unconditionally
            # here -- same as the original code -- but still scoped to just
            # the objects that actually use this datablock, instead of
            # wiping every cached batch in the scene. Live-editing one mesh
            # (e.g. a Skin modifier cage while resizing its radius with
            # Ctrl+A) used to blow away every *other* object's cache too,
            # multiplying how many to_mesh() calls and GPU buffer uploads
            # happened per redraw during a fast-firing modal operator -- the
            # main driver behind crashes while resizing Skin geometry.
            for name in list(gsh_state.batch_cache.keys()) + list(gsh_state.batch_dirty):
                ob = bpy.data.objects.get(name)
                if ob is None:
                    # Object no longer exists -- drop it outright instead of
                    # endlessly re-marking a name that will never resolve again.
                    gsh_state.batch_cache.pop(name, None)
                    gsh_state.batch_dirty.discard(name)
                    gsh_state.batch_last_build.pop(name, None)
                elif getattr(ob, "data", None) == update.id:
                    gsh_state.batch_dirty.add(name)

@persistent
def gsh_on_frame_change(scene, *args, **kwargs):
    # Mark everything dirty rather than dropping it outright: gsh_get_or_build_batch
    # will still redraw the last-known-good silhouette (throttled) while scrubbing
    # the timeline quickly, instead of forcing a synchronous rebuild every frame.
    gsh_state.batch_dirty.update(gsh_state.batch_cache.keys())

def gsh_build_batch_fast(obj, depsgraph):
    if not HAS_NUMPY: return None, None
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = None
    try:
        mesh = eval_obj.to_mesh()
        if mesh is None or len(mesh.vertices) == 0: return None, None
        mesh.calc_loop_triangles()
        n_verts = len(mesh.vertices)
        co = np.empty(n_verts * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        n_tris = len(mesh.loop_triangles)
        if n_tris == 0: return None, None
        tri_idx = np.empty(n_tris * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", tri_idx)
        # Defensive validation before this ever reaches the GPU: Blender's gpu
        # module does not bounds-check GPUIndexBuf/GPUVertBuf data supplied from
        # Python, and a length/range mismatch there is a known source of
        # out-of-bounds GPU reads (i.e. hard crashes, not Python exceptions).
        # Geometry-generating modifiers like Skin can, while their parameters
        # are actively being dragged (e.g. Ctrl+A resize), transiently emit
        # loop triangles faster than this cache can validate them against a
        # matching vertex count, so bail out quietly rather than trust the data.
        if tri_idx.size == 0 or (n_tris * 3) != tri_idx.size:
            return None, None
        if int(tri_idx.min()) < 0 or int(tri_idx.max()) >= n_verts:
            gbfx_state.log_error("gsh_build_batch_fast", ValueError(f"loop_triangles index out of range for {obj.name!r}"))
            return None, None
        tri_idx = tri_idx.reshape(-1, 3)
        local_center = ((co.min(axis=0) + co.max(axis=0)) / 2.0).tolist()
        fmt = gpu.types.GPUVertFormat()
        fmt.attr_add(id="pos", comp_type='F32', len=3, fetch_mode='FLOAT')
        vbo = gpu.types.GPUVertBuf(fmt, n_verts)
        vbo.attr_fill(id="pos", data=co)
        ibo = gpu.types.GPUIndexBuf(type='TRIS', seq=tri_idx)
        batch = gpu.types.GPUBatch(type='TRIS', buf=vbo, elem=ibo)
        return batch, local_center
    except Exception as e:
        gbfx_state.log_error(f"gsh_build_batch_fast ({getattr(obj, 'name', '?')})", e)
        return None, None
    finally:
        if eval_obj is not None and hasattr(eval_obj, 'to_mesh_clear'):
            try: eval_obj.to_mesh_clear()
            except Exception as e: gbfx_state.log_error("gsh_build_batch_fast to_mesh_clear", e)

# Minimum time between forced rebuilds of the same object's GPU batch while it
# keeps getting marked dirty (e.g. a modal operator like Skin resize firing a
# depsgraph update on every mouse-move). Caps how often to_mesh() + a fresh
# VBO/IBO/Batch upload can happen for one object, regardless of how fast the
# underlying modal operator ticks, which both keeps redraws responsive (so
# Active-Tool gizmo interaction stays snappy) and avoids hammering the GPU
# driver with unthrottled allocations -- the main crash risk while live-editing
# geometry-generating modifiers.
GSH_BATCH_REBUILD_MIN_INTERVAL = 0.05

def gsh_get_or_build_batch(obj, depsgraph):
    mode = bpy.context.mode
    if obj == bpy.context.active_object and mode in {'PAINT_VERTEX', 'PAINT_WEIGHT', 'PAINT_TEXTURE'}:
        return gsh_build_batch_fast(obj, depsgraph)

    name = obj.name
    cached = gsh_state.batch_cache.get(name)
    dirty = name in gsh_state.batch_dirty

    if cached is not None and not dirty:
        return cached

    if cached is not None and dirty:
        last_build = gsh_state.batch_last_build.get(name, 0.0)
        if (time.time() - last_build) < GSH_BATCH_REBUILD_MIN_INTERVAL:
            # Still within the throttle window: keep drawing the last-known-good
            # batch this frame rather than rebuilding right now.
            return cached

    entry = gsh_build_batch_fast(obj, depsgraph)
    gsh_state.batch_last_build[name] = time.time()
    gsh_state.batch_dirty.discard(name)
    if entry[0] is not None:
        gsh_state.batch_cache[name] = entry
        return entry
    # Rebuild failed/produced nothing usable (e.g. transient degenerate
    # geometry mid-drag) -- fall back to the last-known-good batch if we have
    # one instead of drawing nothing for a frame.
    return cached if cached is not None else entry

def gsh_build_edit_mode_batches(obj):
    if not HAS_NUMPY: return [], None
    try:
        bm = bmesh.from_edit_mesh(obj.data)
        active_elem = bm.select_history.active
        if not active_elem or not getattr(active_elem, 'select', False): return [], None
        batches = []
        if isinstance(active_elem, bmesh.types.BMFace):
            pos = []
            verts = [v.co[:] for v in active_elem.verts]
            for i in range(1, len(verts) - 1):
                pos.append(verts[0]); pos.append(verts[i]); pos.append(verts[i + 1])
            if pos:
                fmt = gpu.types.GPUVertFormat()
                fmt.attr_add(id="pos", comp_type='F32', len=3, fetch_mode='FLOAT')
                vbo = gpu.types.GPUVertBuf(fmt, len(pos)); vbo.attr_fill(id="pos", data=pos)
                batches.append(('TRIS', gpu.types.GPUBatch(type='TRIS', buf=vbo), None))
        elif isinstance(active_elem, bmesh.types.BMEdge):
            pos = [active_elem.verts[0].co[:], active_elem.verts[1].co[:]]
            fmt = gpu.types.GPUVertFormat(); fmt.attr_add(id="pos", comp_type='F32', len=3, fetch_mode='FLOAT')
            vbo = gpu.types.GPUVertBuf(fmt, len(pos)); vbo.attr_fill(id="pos", data=pos)
            batches.append(('LINES', gpu.types.GPUBatch(type='LINES', buf=vbo), 3.0))
        elif isinstance(active_elem, bmesh.types.BMVert):
            pos = [active_elem.co[:]]
            fmt = gpu.types.GPUVertFormat(); fmt.attr_add(id="pos", comp_type='F32', len=3, fetch_mode='FLOAT')
            vbo = gpu.types.GPUVertBuf(fmt, len(pos)); vbo.attr_fill(id="pos", data=pos)
            batches.append(('POINTS', gpu.types.GPUBatch(type='POINTS', buf=vbo), 8.0))
        else:
            return [], None
        coords = np.array([v.co if hasattr(v, 'co') else v for v in getattr(active_elem, 'verts', [active_elem])], dtype=np.float32)
        sel_center = ((coords.min(axis=0) + coords.max(axis=0)) / 2.0).tolist()
        return batches, sel_center
    except Exception as e:
        # bm.select_history.active and its .co/.verts read directly from the
        # *live* BMesh a modal operator (e.g. Ctrl+A Skin resize) may be
        # actively mutating. That can transiently raise (for example a
        # ReferenceError on an element mid-update) even though nothing is
        # really wrong -- skip this frame's highlight rather than let it
        # propagate out of a GPU draw callback.
        gbfx_state.log_error(f"gsh_build_edit_mode_batches ({getattr(obj, 'name', '?')})", e)
        return [], None

def gsh_get_bone_shape_batch(bone, display_type):
    if not HAS_NUMPY: return None, None, None
    length = getattr(bone, 'length', 1.0)
    if display_type in {'STICK', 'WIRE'}:
        key = ('LINES', round(length, 5))
        if key in gsh_state.bone_batch_cache: return gsh_state.bone_batch_cache[key]
        coords = np.array([[0,0,0], [0, length, 0]], dtype=np.float32)
        fmt = gpu.types.GPUVertFormat(); fmt.attr_add(id="pos", comp_type='F32', len=3, fetch_mode='FLOAT')
        vbo = gpu.types.GPUVertBuf(fmt, 2); vbo.attr_fill(id="pos", data=coords)
        batch = gpu.types.GPUBatch(type='LINES', buf=vbo); res = (batch, 'LINES', 5.0)
        gsh_state.bone_batch_cache[key] = res; return res
    elif display_type == 'BBONE':
        bx = getattr(bone, 'bbone_x', length * 0.1) / 2.0; bz = getattr(bone, 'bbone_z', length * 0.1) / 2.0
        key = ('BBONE', round(length, 5), round(bx, 5), round(bz, 5))
        if key in gsh_state.bone_batch_cache: return gsh_state.bone_batch_cache[key]
        coords = np.array([[-bx, 0, -bz], [ bx, 0, -bz], [ bx, 0,  bz], [-bx, 0,  bz], [-bx, length, -bz], [ bx, length, -bz], [ bx, length,  bz], [-bx, length,  bz]], dtype=np.float32)
        indices = np.array([[0,1,2], [0,2,3], [4,6,5], [4,7,6], [0,4,5], [0,5,1], [1,5,6], [1,6,2], [2,6,7], [2,7,3], [3,7,4], [3,4,0]], dtype=np.int32)
        fmt = gpu.types.GPUVertFormat(); fmt.attr_add(id="pos", comp_type='F32', len=3, fetch_mode='FLOAT')
        vbo = gpu.types.GPUVertBuf(fmt, 8); vbo.attr_fill(id="pos", data=coords); ibo = gpu.types.GPUIndexBuf(type='TRIS', seq=indices)
        batch = gpu.types.GPUBatch(type='TRIS', buf=vbo, elem=ibo); res = (batch, 'TRIS', None)
        gsh_state.bone_batch_cache[key] = res; return res
    elif display_type == 'ENVELOPE':
        hr = getattr(bone, 'head_radius', length * 0.05); tr = getattr(bone, 'tail_radius', length * 0.05)
        key = ('ENVELOPE', round(length, 5), round(hr, 5), round(tr, 5))
        if key in gsh_state.bone_batch_cache: return gsh_state.bone_batch_cache[key]
        coords = np.array([[-hr, 0, -hr], [ hr, 0, -hr], [ hr, 0,  hr], [-hr, 0,  hr], [-tr, length, -tr], [ tr, length, -tr], [ tr, length,  tr], [-tr, length,  tr]], dtype=np.float32)
        indices = np.array([[0,1,2], [0,2,3], [4,6,5], [4,7,6], [0,4,5], [0,5,1], [1,5,6], [1,6,2], [2,6,7], [2,7,3], [3,7,4], [3,4,0]], dtype=np.int32)
        fmt = gpu.types.GPUVertFormat(); fmt.attr_add(id="pos", comp_type='F32', len=3, fetch_mode='FLOAT')
        vbo = gpu.types.GPUVertBuf(fmt, 8); vbo.attr_fill(id="pos", data=coords); ibo = gpu.types.GPUIndexBuf(type='TRIS', seq=indices)
        batch = gpu.types.GPUBatch(type='TRIS', buf=vbo, elem=ibo); res = (batch, 'TRIS', None)
        gsh_state.bone_batch_cache[key] = res; return res
    else: 
        key = ('OCTA', round(length, 5))
        if key in gsh_state.bone_batch_cache: return gsh_state.bone_batch_cache[key]
        w = length * 0.1; f = length * 0.1
        coords = np.array([[0, 0, 0], [0, length, 0], [w, f, 0], [-w, f, 0], [0, f, w], [0, f, -w]], dtype=np.float32)
        indices = np.array([[0, 2, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2], [1, 4, 2], [1, 3, 4], [1, 5, 3], [1, 2, 5]], dtype=np.int32)
        fmt = gpu.types.GPUVertFormat(); fmt.attr_add(id="pos", comp_type='F32', len=3, fetch_mode='FLOAT')
        vbo = gpu.types.GPUVertBuf(fmt, 6); vbo.attr_fill(id="pos", data=coords); ibo = gpu.types.GPUIndexBuf(type='TRIS', seq=indices)
        batch = gpu.types.GPUBatch(type='TRIS', buf=vbo, elem=ibo); res = (batch, 'TRIS', None)
        gsh_state.bone_batch_cache[key] = res; return res

def gsh_build_active_bone_batches(context, depsgraph):
    obj = context.active_object
    if not obj or obj.type != 'ARMATURE': return [], None
    armature = obj.data; display_type = getattr(armature, 'display_type', 'OCTAHEDRAL'); batches = []; centers = []
    def process_pbone(pbone):
        shape_batch = None; prim_type = 'TRIS'; extra = None; mat = obj.matrix_world @ pbone.matrix
        custom_shape = getattr(pbone, 'custom_shape', None); center = None
        if custom_shape and custom_shape.type == 'MESH':
            shape_batch, _lc = gsh_get_or_build_batch(custom_shape, depsgraph)
            if shape_batch:
                c_mat = pbone.matrix.copy(); transform_bone = getattr(pbone, 'custom_shape_transform', None)
                if transform_bone: c_mat = transform_bone.matrix.copy()
                if getattr(pbone, 'use_custom_shape_bone_size', True): c_mat = c_mat @ Matrix.Scale(pbone.bone.length, 4)
                trans = getattr(pbone, 'custom_shape_translation', (0,0,0)); rot = getattr(pbone, 'custom_shape_rotation_euler', (0,0,0)); scale = getattr(pbone, 'custom_shape_scale_xyz', (1,1,1))
                loc_mat = Matrix.Translation(trans); rot_mat = Euler(rot, 'XYZ').to_matrix().to_4x4(); scale_mat = Matrix.Diagonal(Vector(scale).to_4d())
                c_mat = c_mat @ loc_mat @ rot_mat @ scale_mat; mat = obj.matrix_world @ c_mat
                center = mat @ Vector(_lc) if _lc else mat @ Vector((0, 0, 0))
        else:
            shape_batch, prim_type, extra = gsh_get_bone_shape_batch(pbone.bone, display_type)
        if shape_batch:
            batches.append((prim_type, shape_batch, mat, extra))
            if center is None: center = obj.matrix_world @ pbone.matrix @ Vector((0, pbone.bone.length * 0.5, 0))
            centers.append(center)
            
    if context.mode == 'POSE':
        pbone = context.active_pose_bone
        if pbone: process_pbone(pbone)
    elif context.mode == 'EDIT_ARMATURE':
        ebone = context.active_bone
        if ebone:
            shape_batch, prim_type, extra = gsh_get_bone_shape_batch(ebone, display_type)
            if shape_batch:
                mat = obj.matrix_world @ ebone.matrix; batches.append((prim_type, shape_batch, mat, extra)); centers.append(mat @ Vector((0, ebone.length * 0.5, 0)))
    elif context.mode == 'OBJECT':
        for pbone in obj.pose.bones: process_pbone(pbone)
    if not batches: return [], None
    sel_center = sum(centers, Vector((0,0,0))) / len(centers)
    return batches, sel_center

def gsh__project_point(view_proj, region, world_co):
    p = view_proj @ Vector((world_co.x, world_co.y, world_co.z, 1.0))
    if p.w <= 1e-6: return None
    x = (p.x / p.w * 0.5 + 0.5) * region.width; y = (p.y / p.w * 0.5 + 0.5) * region.height
    return (x, y)

class gsh__ScreenRect:
    __slots__ = ("xmin", "ymin", "xmax", "ymax")
    def __init__(self, xmin, ymin, xmax, ymax): self.xmin, self.ymin, self.xmax, self.ymax = xmin, ymin, xmax, ymax
    def overlaps(self, other):
        return not (self.xmax < other.xmin or other.xmax < self.xmin or self.ymax < other.ymin or other.ymax < self.ymin)

def gsh_screen_bbox(obj, view_proj, region):
    pts = []
    for corner in obj.bound_box:
        wc = obj.matrix_world @ Vector(corner); sp = gsh__project_point(view_proj, region, wc)
        if sp is not None: pts.append(sp)
    if not pts: return None
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return gsh__ScreenRect(min(xs), min(ys), max(xs), max(ys))

def gsh_get_occluder_candidates(context, active_objs, view_proj, region, rv3d):
    active_set = set(active_objs)
    active_rects = [r for r in (gsh_screen_bbox(o, view_proj, region) for o in active_objs) if r is not None]
    if not active_rects: return []
    result = []
    cam_pos = rv3d.view_matrix.inverted().translation if rv3d else None
    
    for obj in context.visible_objects:
        if obj in active_set or obj.type != 'MESH' or getattr(obj, "show_in_front", False): continue
        
        # Coarse culling: ignore objects behind the camera
        if cam_pos:
            to_obj = obj.location - cam_pos
            view_dir = rv3d.view_matrix[2][:3]
            if Vector(view_dir).dot(to_obj) > 0: 
                continue
                
        rect = gsh_screen_bbox(obj, view_proj, region)
        if rect is None: continue
        if any(rect.overlaps(ar) for ar in active_rects): result.append(obj)
    return result

def gsh__expand_bounds(pmin, pmax, point):
    if pmin is None: return Vector(point), Vector(point)
    pmin = Vector((min(pmin.x, point.x), min(pmin.y, point.y), min(pmin.z, point.z)))
    pmax = Vector((max(pmax.x, point.x), max(pmax.y, point.y), max(pmax.z, point.z)))
    return pmin, pmax


def gsh__get_outline_offscreen(region_id, width, height):
    entry = gsh_state.outline_offscreens.get(region_id)
    if entry is None or entry[1] != (width, height):
        if entry is not None:
            try: entry[0].free()
            except Exception as e: gbfx_state.log_error("_get_outline_offscreen free", e)
        offscreen = gpu.types.GPUOffScreen(width, height)
        gsh_state.outline_offscreens[region_id] = (offscreen, (width, height))
        return offscreen
    return entry[0]

def gsh__free_outline_offscreens():
    for offscreen, _size in gsh_state.outline_offscreens.values():
        try: offscreen.free()
        except Exception as e: gbfx_state.log_error("_free_outline_offscreens", e)
    gsh_state.outline_offscreens.clear(); gsh_state.bone_batch_cache.clear()

def gsh_render_mask(offscreen, active_batches, occluder_batches, view_proj, depth_test):
    depth_shader = gsh__get_outline_depth_shader(); mask_shader = gsh__get_outline_mask_shader()
    with offscreen.bind():
        fb = gpu.state.active_framebuffer_get()
        fb.clear(color=(0.0, 0.0, 0.0, 0.0), depth=1.0)
        try:
            if depth_test and occluder_batches:
                gpu.state.depth_test_set("LESS_EQUAL"); gpu.state.color_mask_set(False, False, False, False); gpu.state.depth_mask_set(True)
                depth_shader.bind(); depth_shader.uniform_float("viewProjectionMatrix", view_proj)
                for batch, model_matrix in occluder_batches:
                    depth_shader.uniform_float("modelMatrix", model_matrix); batch.draw(depth_shader)
                gpu.state.color_mask_set(True, True, True, True); gpu.state.depth_mask_set(False)
            else:
                gpu.state.depth_test_set("NONE"); gpu.state.color_mask_set(True, True, True, True); gpu.state.depth_mask_set(True)
            mask_shader.bind(); mask_shader.uniform_float("viewProjectionMatrix", view_proj)
            for prim_type, batch, model_matrix, extra in active_batches:
                mask_shader.uniform_float("modelMatrix", model_matrix)
                if prim_type == 'LINES':
                    gpu.state.line_width_set(extra or 3.0); batch.draw(mask_shader); gpu.state.line_width_set(1.0)
                elif prim_type == 'POINTS':
                    gpu.state.point_size_set(extra or 8.0); batch.draw(mask_shader); gpu.state.point_size_set(1.0)
                else: batch.draw(mask_shader)
        finally:
            # Always land back on a known-good pipeline state, even if a
            # batch.draw() call above raised partway through. Leaving e.g.
            # color writes disabled (from the occluder pre-pass) would silently
            # break *everything* drawn afterwards in this viewport for the rest
            # of the frame -- including Blender's own gizmo/overlay drawing.
            gpu.state.depth_test_set("NONE"); gpu.state.depth_mask_set(True); gpu.state.color_mask_set(True, True, True, True)
            gpu.state.line_width_set(1.0); gpu.state.point_size_set(1.0)

def gsh__composite_outline(offscreen, region, pivot_screen, prefs):
    shader = gsh__get_outline_compose_shader()
    width, height = region.width, region.height
    verts = [(0, 0), (width, 0), (width, height), (0, height)]
    indices = [(0, 1, 2), (2, 3, 0)]
    batch = batch_for_shader(shader, "TRIS", {"pos": verts}, indices=indices)
    colors = [tuple(c) for c in gbfx_colors(prefs)]
    try:
        gpu.state.depth_test_set("NONE"); gpu.state.blend_set("ALPHA")
        shader.bind()
        shader.uniform_sampler("mask_tex", offscreen.texture_color) 
        shader.uniform_float("u_resolution", (width, height)); shader.uniform_float("texel_size", (1.0 / width, 1.0 / height))
        shader.uniform_float("outline_px", prefs.gsh_viewport_outline_width); shader.uniform_float("pivot_screen", pivot_screen)
        shader.uniform_float("phase01", gsh_state.outline_rot)
        shader.uniform_float("color0", colors[0]); shader.uniform_float("color1", colors[1]); shader.uniform_float("color2", colors[2]); shader.uniform_float("color3", colors[3])
        batch.draw(shader)
    finally:
        # Same reasoning as gsh_render_mask: always restore blend/depth state
        # to Blender's expected defaults for the rest of the viewport draw,
        # even if the composite draw call above failed.
        gpu.state.blend_set("NONE"); gpu.state.depth_test_set("LESS_EQUAL")

def gsh__draw_node_outline():
    try:
        context = bpy.context
        prefs = get_prefs(context)
        if not prefs or not prefs.gsh_node_outline_enabled: return
        space = context.space_data
        if space is None or space.type != 'NODE_EDITOR': return
        if not space.edit_tree: return
        node = space.edit_tree.nodes.active
        if not node or not getattr(node, "select", True): return
        region = context.region
        view2d = region.view2d
        if not view2d: return
        loc = node.location.copy(); parent = node.parent
        while parent: loc += parent.location; parent = parent.parent
        dim = node.dimensions
        if dim.x < 1 or dim.y < 1: return
        ui_scale = context.preferences.system.ui_scale
        loc_x, loc_y = loc.x * ui_scale, loc.y * ui_scale
        p1_x, p1_y = loc_x, loc_y
        p2_x, p2_y = loc_x + dim.x, loc_y
        p3_x, p3_y = loc_x + dim.x, loc_y - dim.y
        p4_x, p4_y = loc_x, loc_y - dim.y
        p1 = view2d.view_to_region(p1_x, p1_y, clip=False)
        p2 = view2d.view_to_region(p2_x, p2_y, clip=False)
        p3 = view2d.view_to_region(p3_x, p3_y, clip=False)
        p4 = view2d.view_to_region(p4_x, p4_y, clip=False)
        if not all((p1, p2, p3, p4)): return
        clamp_x, clamp_y = region.width * 4, region.height * 4
        def _clamp_pt(p): return (max(-clamp_x, min(clamp_x, p[0])), max(-clamp_y, min(clamp_y, p[1])))
        p1, p2, p3, p4 = _clamp_pt(p1), _clamp_pt(p2), _clamp_pt(p3), _clamp_pt(p4)
        pad = prefs.gsh_viewport_outline_width + 4.0
        polygon = [(p1[0] - pad, p1[1] + pad), (p2[0] + pad, p2[1] + pad), (p3[0] + pad, p3[1] - pad), (p4[0] - pad, p4[1] - pad)]
        lw = prefs.gsh_viewport_outline_width
        gsh__draw_rotating_outline(polygon, gsh_state.outline_rot, gbfx_colors(prefs), line_width=lw)
    except Exception as e: gbfx_state.log_error("_draw_node_outline", e)

def gsh_draw_viewport_outline():
    try:
        _gsh_draw_viewport_outline_impl()
    except Exception as e:
        gbfx_state.log_error("gsh_draw_viewport_outline", e)
        # Best-effort safety net: make sure nothing later in this frame -- most
        # importantly Blender's own gizmo/overlay drawing -- inherits a broken
        # pipeline state (disabled color writes, stuck blend mode, etc.) left
        # behind by whichever step above failed before it could restore things
        # itself. This is on top of (not a replacement for) the try/finally
        # blocks inside gsh_render_mask/gsh__composite_outline.
        try:
            gpu.state.color_mask_set(True, True, True, True)
            gpu.state.depth_mask_set(True)
            gpu.state.depth_test_set("LESS_EQUAL")
            gpu.state.blend_set("NONE")
            gpu.state.line_width_set(1.0)
            gpu.state.point_size_set(1.0)
        except Exception:
            pass

def _gsh_draw_viewport_outline_impl():
    context = bpy.context
    prefs = get_prefs(context)
    if not prefs or not prefs.gsh_viewport_outline_enabled: return
    if context.mode == 'SCULPT': return
    space = context.space_data
    if space and getattr(space, 'type', '') == 'VIEW_3D':
        if not getattr(space.overlay, "show_overlays", True): return
    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None or region.type != "WINDOW": return
    
    obj = context.active_object
    if not obj or not obj.visible_get(): return

    view_proj = rv3d.perspective_matrix.copy()
    depsgraph = context.evaluated_depsgraph_get()
    active_batches = []
    pivot_min = pivot_max = None
    occluder_objs = []
    
    depth_test = prefs.gsh_viewport_outline_depth_test
    if getattr(obj, "show_in_front", False):
        depth_test = False
        
    if context.mode in {'POSE', 'EDIT_ARMATURE'}:
        bone_batches, world_center = gsh_build_active_bone_batches(context, depsgraph)
        if not bone_batches: return
        for prim_type, batch, model_matrix, extra in bone_batches: active_batches.append((prim_type, batch, model_matrix, extra))
        if world_center is not None: pivot_min, pivot_max = gsh__expand_bounds(pivot_min, pivot_max, world_center)
        if depth_test: occluder_objs = gsh_get_occluder_candidates(context, [context.active_object], view_proj, region, rv3d)
    elif context.mode == 'EDIT_MESH' and prefs.gsh_viewport_outline_edit_mode:
        obj_edit = context.edit_object
        if obj_edit is None or obj_edit.type != 'MESH' or not obj_edit.select_get() or not obj_edit.visible_get(): return
        edit_batches, sel_center = gsh_build_edit_mode_batches(obj_edit)
        if not edit_batches: return
        model_matrix = obj_edit.matrix_world.copy()
        for prim_type, batch, extra in edit_batches: active_batches.append((prim_type, batch, model_matrix, extra))
        if sel_center is not None:
            wc = obj_edit.matrix_world @ Vector(sel_center)
            pivot_min, pivot_max = gsh__expand_bounds(pivot_min, pivot_max, wc)
        if depth_test: occluder_objs = gsh_get_occluder_candidates(context, [obj_edit], view_proj, region, rv3d)
    else:
        if not obj.select_get(): return
        if obj.type in {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT'}:
            batch, local_center = gsh_get_or_build_batch(obj, depsgraph)
            if batch is None: return
            active_batches.append(('TRIS', batch, obj.matrix_world.copy(), None))
            if local_center is not None:
                wc = obj.matrix_world @ Vector(local_center)
                pivot_min, pivot_max = gsh__expand_bounds(pivot_min, pivot_max, wc)
        elif obj.type == 'ARMATURE':
            bone_batches, world_center = gsh_build_active_bone_batches(context, depsgraph)
            if not bone_batches: return
            for prim_type, batch, model_matrix, extra in bone_batches: active_batches.append((prim_type, batch, model_matrix, extra))
            if world_center is not None: pivot_min, pivot_max = gsh__expand_bounds(pivot_min, pivot_max, world_center)
        else: return
        if depth_test: occluder_objs = gsh_get_occluder_candidates(context, [obj], view_proj, region, rv3d)

    occluder_batches = []
    if depth_test:
        for o in occluder_objs:
            if o.type == 'MESH':
                batch, _lc = gsh_get_or_build_batch(o, depsgraph)
                if batch is not None: occluder_batches.append((batch, o.matrix_world.copy()))

    scale = max(0.25, min(1.0, prefs.gsh_viewport_outline_resolution))
    w = max(4, int(region.width * scale))
    h = max(4, int(region.height * scale))
    offscreen = gsh__get_outline_offscreen(region.as_pointer(), w, h)
    gsh_render_mask(offscreen, active_batches, occluder_batches, view_proj, depth_test)
    if pivot_min is None: pivot_screen = (region.width / 2.0, region.height / 2.0)
    else:
        pivot_world = (pivot_min + pivot_max) / 2.0
        pivot_screen = gsh__project_point(view_proj, region, pivot_world) or (region.width / 2.0, region.height / 2.0)
    gsh__composite_outline(offscreen, region, pivot_screen, prefs)

def gsh__draw_activity_border_image():
    try:
        context = bpy.context; prefs = get_prefs(context)
        if not prefs or not prefs.gsh_border_enabled: return
        space = context.space_data
        if space is None or space.image is None or space.image.type != "RENDER_RESULT": return
        region = context.region
        if region is None or region.type != "WINDOW": return
        lw = prefs.gsh_border_line_width
        inset = lw / 2.0  
        polygon = [(inset, inset), (region.width - inset, inset), (region.width - inset, region.height - inset), (inset, region.height - inset)]
        gsh__draw_rotating_outline(polygon, gsh_state.border_rot, gbfx_colors(prefs), line_width=lw)
    except Exception as e: gbfx_state.log_error("_draw_activity_border_image", e)

# =============================================================================
# UNIFIED MASTER TIMER & TOGGLES
# =============================================================================

def gsh_master_tick():
    """Unified timer for all Chaos-family rotating effects."""
    prefs = get_prefs(bpy.context)
    if not prefs or not gsh_state.master_timer_running: return None

    interval = 0.03
    areas_to_redraw = set()

    if prefs.gsh_chaos_enabled:
        gsh_state.chaos_rot = (gsh_state.chaos_rot + interval / max(prefs.gsh_chaos_speed, 0.15)) % 1.0
        
        if gsh_state.chaos_targets:
            colors = gbfx_colors(prefs); n = len(colors); idx = gsh_state.chaos_rot * n
            i1, i2 = int(idx) % n, (int(idx) + 1) % n
            t = idx % 1.0; t = t * t * (3.0 - 2.0 * t) 
            c1, c2 = colors[i1], colors[i2]
            r, g, b = c1[0]+(c2[0]-c1[0])*t, c1[1]+(c2[1]-c1[1])*t, c1[2]+(c2[2]-c1[2])*t
            
            for obj, prop_id, orig_r, orig_g, orig_b, orig_a, size in gsh_state.chaos_targets:
                try: setattr(obj, prop_id, (r, g, b) if size==3 else (r, g, b, orig_a))
                except Exception: pass
            areas_to_redraw.update({'VIEW_3D', 'NODE_EDITOR', 'OUTLINER', 'PROPERTIES'})

    if prefs.gsh_viewport_outline_enabled:
        gsh_state.outline_rot = (gsh_state.outline_rot + interval / max(prefs.gsh_viewport_outline_speed, 0.15)) % 1.0
        areas_to_redraw.add('VIEW_3D')
        
    if prefs.gsh_node_outline_enabled:
        gsh_state.outline_rot = (gsh_state.outline_rot + interval / max(prefs.gsh_viewport_outline_speed, 0.15)) % 1.0
        areas_to_redraw.add('NODE_EDITOR')

    if prefs.gsh_border_enabled and getattr(bpy.context.space_data, "type", "") == "IMAGE_EDITOR":
        gsh_state.border_rot = (gsh_state.border_rot + interval / max(prefs.gsh_border_speed, 0.15)) % 1.0
        areas_to_redraw.add('IMAGE_EDITOR')

    if areas_to_redraw:
        gbfx_tag_redraw(areas_to_redraw)

    return interval

def gsh_evaluate_state(context=None):
    """Centralized manager for registering/unregistering handles and timers."""
    prefs = get_prefs(context)
    if not prefs: return

    # Theme Animator
    if prefs.gsh_chaos_enabled and not gsh_state.chaos_active:
        gsh_state.chaos_active = True
        gsh_rescan_theme(prefs)
    elif not prefs.gsh_chaos_enabled and gsh_state.chaos_active:
        gsh_state.chaos_active = False
        gsh_restore_theme()

    # Viewport Outline
    if prefs.gsh_viewport_outline_enabled and not gsh_state.vp_handle:
        gsh_state.vp_handle = bpy.types.SpaceView3D.draw_handler_add(gsh_draw_viewport_outline, (), "WINDOW", "POST_PIXEL")
    elif not prefs.gsh_viewport_outline_enabled and gsh_state.vp_handle:
        bpy.types.SpaceView3D.draw_handler_remove(gsh_state.vp_handle, "WINDOW")
        gsh_state.vp_handle = None
        gsh__free_outline_offscreens()

    # Node Outline
    if prefs.gsh_node_outline_enabled and not gsh_state.node_handle:
        gsh_state.node_handle = bpy.types.SpaceNodeEditor.draw_handler_add(gsh__draw_node_outline, (), "WINDOW", "POST_PIXEL")
    elif not prefs.gsh_node_outline_enabled and gsh_state.node_handle:
        bpy.types.SpaceNodeEditor.draw_handler_remove(gsh_state.node_handle, "WINDOW")
        gsh_state.node_handle = None

    # Timer Management
    any_enabled = prefs.gsh_chaos_enabled or prefs.gsh_viewport_outline_enabled or prefs.gsh_node_outline_enabled or prefs.gsh_border_enabled
    if any_enabled and not gsh_state.master_timer_running:
        gsh_state.master_timer_running = True
        bpy.app.timers.register(gsh_master_tick, first_interval=0.0)
    elif not any_enabled and gsh_state.master_timer_running:
        gsh_state.master_timer_running = False

    gbfx_tag_redraw({'VIEW_3D', 'NODE_EDITOR', 'UI'})


class GUSHING_OT_toggle_chaos(Operator):
    bl_idname = "gbfx.gsh_toggle_chaos"
    bl_label = "Toggle Magical Girls Chaos"
    def execute(self, context):
        prefs = get_prefs(context)
        prefs.gsh_chaos_enabled = not prefs.gsh_chaos_enabled
        self.report({'INFO'}, "Theme Animator " + ("Enabled" if prefs.gsh_chaos_enabled else "Disabled"))
        return {"FINISHED"}

class GUSHING_OT_toggle_viewport_outline(Operator):
    bl_idname = "gbfx.gsh_toggle_viewport_outline"
    bl_label = "Toggle Viewport Silhouette Outline"
    def execute(self, context):
        prefs = get_prefs(context)
        prefs.gsh_viewport_outline_enabled = not prefs.gsh_viewport_outline_enabled
        self.report({'INFO'}, "Viewport Outline " + ("Enabled" if prefs.gsh_viewport_outline_enabled else "Disabled"))
        return {"FINISHED"}

class GUSHING_OT_toggle_node_outline(Operator):
    bl_idname = "gbfx.gsh_toggle_node_outline"
    bl_label = "Toggle Active Node Outline"
    def execute(self, context):
        prefs = get_prefs(context)
        prefs.gsh_node_outline_enabled = not prefs.gsh_node_outline_enabled
        self.report({'INFO'}, "Node Outline " + ("Enabled" if prefs.gsh_node_outline_enabled else "Disabled"))
        return {"FINISHED"}

class GUSHING_OT_toggle_border(Operator):
    bl_idname = "gbfx.gsh_toggle_border"
    bl_label = "Toggle Render/Bake Activity Border"
    def execute(self, context):
        prefs = get_prefs(context)
        prefs.gsh_border_enabled = not prefs.gsh_border_enabled
        self.report({'INFO'}, "Render Border " + ("Enabled" if prefs.gsh_border_enabled else "Disabled"))
        return {"FINISHED"}


class GUSHING_OT_load_theme_xml(Operator):
    bl_idname = "gbfx.gsh_load_theme_xml"
    bl_label = "Load Colors From Theme XML"
    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.xml", options={"HIDDEN"})
    
    def execute(self, context):
        prefs = get_prefs(context)
        if not prefs: return {"CANCELLED"}
        if not os.path.isfile(self.filepath): self.report({"ERROR"}, "File not found"); return {"CANCELLED"}
        try: tree = ET.parse(self.filepath)
        except ET.ParseError as e: self.report({"ERROR"}, f"Could not parse XML: {e}"); return {"CANCELLED"}
        
        root = tree.getroot()
        ui = root.find(".//user_interface//ThemeUserInterface")
        if ui is not None:
            found = []
            for attr in ("axis_x", "axis_z", "axis_w", "axis_y"):
                if attr in ui.attrib: found.append(srgb_hex_to_linear(ui.attrib[attr]))
            if len(found) >= 4:
                prefs.gbfx_color_1 = found[0]
                prefs.gbfx_color_2 = found[1]
                prefs.gbfx_color_3 = found[2]
                prefs.gbfx_color_4 = found[3]
                self.report({"INFO"}, "Theme colors loaded successfully")
                gsh_rescan_if_enabled()
                return {"FINISHED"}
            else:
                self.report({"WARNING"}, f"Found {len(found)}/4 required colors in XML. Aborted.")
                return {"CANCELLED"}
        self.report({"WARNING"}, "No UI theme data found in XML.")
        return {"CANCELLED"}
        
    def invoke(self, context, event): context.window_manager.fileselect_add(self); return {"RUNNING_MODAL"}

def gsh_border_start():
    prefs = get_prefs(bpy.context)
    if not prefs or not prefs.gsh_border_enabled: return
    if gsh_state.border_handle is None: gsh_state.border_handle = bpy.types.SpaceImageEditor.draw_handler_add(gsh__draw_activity_border_image, (), "WINDOW", "POST_PIXEL")
    gsh_evaluate_state()

def gsh_border_stop():
    if gsh_state.border_handle:
        bpy.types.SpaceImageEditor.draw_handler_remove(gsh_state.border_handle, "WINDOW")
        gsh_state.border_handle = None
    gbfx_tag_redraw({'IMAGE_EDITOR'})


def gsh__on_render_init(scene, depsgraph=None): gsh_border_start()
def gsh__on_render_done(scene, depsgraph=None): gsh_border_stop()


# =============================================================================
# MAIN UI PANEL
# =============================================================================

class GBFX_PT_main_panel(Panel):
    bl_label = "Gushing FX"
    bl_idname = "GBFX_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Gushing FX"

    def draw(self, context):
        layout = self.layout
        prefs = get_prefs(context)
        
        if not HAS_NUMPY:
            box = layout.box()
            box.alert = True
            box.label(text="Warning: 'numpy' module missing!", icon='ERROR')
            box.label(text="Outlines and Edit/Pose mode FX will not work.")
            
        if gbfx_state.errors:
            box = layout.box()
            box.alert = True
            box.label(text="Recent Runtime Errors:", icon='ERROR')
            for err in gbfx_state.errors:
                box.label(text=err[:50] + "..." if len(err) > 50 else err)
            box.operator("gbfx.clear_errors", icon='TRASH')
            
        if not prefs:
            layout.label(text="Preferences not ready", icon='ERROR')
            return

        col = layout.column(align=True)
        col.prop(prefs, "mb_enable_fx", text="Magia Baiser FX", toggle=True, icon='SHADERFX')
        if prefs.mb_enable_fx:
            row = col.row(align=True)
            row.prop(prefs, "mb_preview_effect", text="")
            row.operator("gbfx.mb_preview_effect", text="Preview", icon='PLAY')
            col.operator("gbfx.mb_apply_all_modifiers", icon='MODIFIER')

        layout.separator()
        col = layout.column(align=True)
        col.prop(prefs, "ht_enabled", text="Halftone Overlay", toggle=True, icon='SHADING_RENDERED')
        if prefs.ht_enabled:
            box = col.box()
            box.prop(prefs, "ht_dot_color", text="Color")
            box.prop(prefs, "ht_opacity", text="Opacity")
            box.prop(prefs, "ht_dot_density", text="Density")
            box.prop(prefs, "ht_max_dot_size", text="Size")
            box.prop(prefs, "ht_pattern_rotation", text="Rotation")

        layout.separator()
        col = layout.column(align=True)
        col.prop(prefs, "uto_enabled", text="UI Text Outline", toggle=True, icon='USER')
        if prefs.uto_enabled:
            col.operator("gbfx.uto_refresh", icon='FILE_REFRESH', text="Refresh UI")

        layout.separator()
        box = layout.box()
        box.label(text="Gushing Chaos", icon='COLOR')
        col = box.column(align=True)
        col.operator("gbfx.gsh_toggle_chaos", text="Theme Animator: " + ("On" if prefs.gsh_chaos_enabled else "Off"), depress=prefs.gsh_chaos_enabled, icon="COLOR")
        col.operator("gbfx.gsh_toggle_viewport_outline", text="Viewport Silhouette Outline: " + ("On" if prefs.gsh_viewport_outline_enabled else "Off"), depress=prefs.gsh_viewport_outline_enabled, icon="MOD_OUTLINE")
        col.operator("gbfx.gsh_toggle_node_outline", text="Node Editor Outline: " + ("On" if prefs.gsh_node_outline_enabled else "Off"), depress=prefs.gsh_node_outline_enabled, icon="NODETREE")
        col.operator("gbfx.gsh_toggle_border", text="Bake/Render Border: " + ("On" if prefs.gsh_border_enabled else "Off"), depress=prefs.gsh_border_enabled, icon="SEQUENCE")
        
        if prefs.gsh_viewport_outline_enabled or prefs.gsh_node_outline_enabled:
            b = box.box()
            b.prop(prefs, "gsh_viewport_outline_width")
            b.prop(prefs, "gsh_viewport_outline_speed")

        box.separator()
        box.operator("gbfx.gsh_load_theme_xml", icon="FILE_FOLDER")

        layout.separator()
        layout.label(text="Full settings in Add-on Preferences", icon='PREFERENCES')


# =============================================================================
# REGISTRATION & LIFECYCLE
# =============================================================================

classes = (
    GBFX_Preferences,
    GBFX_OT_clear_errors,
    MBFX_OT_apply_all_modifiers,
    MBFX_OT_preview_effect,
    UITEXTOUTLINE_OT_refresh,
    HALFTONE_OT_reset_settings,
    GUSHING_OT_load_theme_xml,
    GUSHING_OT_toggle_chaos,
    GUSHING_OT_rescan_theme,
    GUSHING_OT_toggle_viewport_outline,
    GUSHING_OT_toggle_node_outline,
    GUSHING_OT_toggle_border,
    GBFX_PT_main_panel,
)

@persistent
def gbfx__on_file_load(scene, depsgraph=None):
    try:
        if gsh_state.vp_handle is not None: bpy.types.SpaceView3D.draw_handler_remove(gsh_state.vp_handle, "WINDOW")
        if gsh_state.node_handle is not None: bpy.types.SpaceNodeEditor.draw_handler_remove(gsh_state.node_handle, "WINDOW")
        if gsh_state.border_handle is not None: bpy.types.SpaceImageEditor.draw_handler_remove(gsh_state.border_handle, "WINDOW")
    except Exception as e: gbfx_state.log_error("_on_file_load remove handlers", e)

    gsh_state.master_timer_running = False
    gsh_state.chaos_active = False
    gsh_state.vp_handle = None
    gsh_state.node_handle = None
    gsh_state.border_handle = None
    gsh_state.batch_cache.clear()
    gsh_state.batch_dirty.clear()
    gsh_state.batch_last_build.clear()
    gsh__free_outline_offscreens()
    
    bpy.app.timers.register(gbfx__autostart_effects, first_interval=0.5)

def gbfx__autostart_effects():
    gsh_evaluate_state()
    mb__delayed_startup()
    return None

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    # 1. Magia Baiser Init
    mb_state.running = True
    mb_state.effects = []
    mb_state.obj_count = 0
    mb_state.obj_cache = {}
    mb_state.suppress_until = 0.0
    mb_state.last_modifier_op_id = None
    mb_state.last_selection_hash = None
    mb_state.handle_view = bpy.types.SpaceView3D.draw_handler_add(mb_draw_callback_view, (), 'WINDOW', 'POST_VIEW')
    mb_state.handle_pixel = bpy.types.SpaceView3D.draw_handler_add(mb_draw_callback_pixel, (), 'WINDOW', 'POST_PIXEL')
    
    if mb_on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post: bpy.app.handlers.depsgraph_update_post.append(mb_on_depsgraph_update)
    if mb_on_undo_post not in bpy.app.handlers.undo_post: bpy.app.handlers.undo_post.append(mb_on_undo_post)
    if mb_on_redo_post not in bpy.app.handlers.redo_post: bpy.app.handlers.redo_post.append(mb_on_redo_post)
    if mb_on_save_post not in bpy.app.handlers.save_post: bpy.app.handlers.save_post.append(mb_on_save_post)
    if mb_on_load_post not in bpy.app.handlers.load_post: bpy.app.handlers.load_post.append(mb_on_load_post)

    mb__subscribe_msgbus()
    bpy.app.timers.register(mb_fx_tick, persistent=True)
    bpy.app.timers.register(mb_selection_poll_tick, persistent=True)

    # 2. UI Text Outline Init
    if uto__on_load_post not in bpy.app.handlers.load_post: bpy.app.handlers.load_post.append(uto__on_load_post)
    # Deferred call to prevent _RestrictData crashes during Addon boot
    bpy.app.timers.register(lambda: (uto_apply_outline(), None)[1], first_interval=0.1)

    # 3. Halftone Init
    if not bpy.app.background:
        if ht_state.handle is None: ht_state.handle = bpy.types.SpaceView3D.draw_handler_add(ht_draw_callback, (), 'WINDOW', 'POST_VIEW')

    # 4. Gushing Chaos Init
    bpy.app.timers.register(gbfx__autostart_effects, first_interval=0.5)
    if hasattr(bpy.app.handlers, "load_post"): bpy.app.handlers.load_post.append(gbfx__on_file_load)
    
    if gsh_outline_on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post: 
        bpy.app.handlers.depsgraph_update_post.append(gsh_outline_on_depsgraph_update)
        
    if gsh_on_frame_change not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(gsh_on_frame_change)
        
    if hasattr(bpy.app.handlers, "render_init"): bpy.app.handlers.render_init.append(gsh__on_render_init)
    bpy.app.handlers.render_complete.append(gsh__on_render_done)
    bpy.app.handlers.render_cancel.append(gsh__on_render_done)
    if hasattr(bpy.app.handlers, "object_bake_pre"):
        bpy.app.handlers.object_bake_pre.append(gsh__on_render_init)
        bpy.app.handlers.object_bake_complete.append(gsh__on_render_done)
        bpy.app.handlers.object_bake_cancel.append(gsh__on_render_done)

def unregister():
    # 1. Magia Baiser cleanup
    mb_state.running = False
    if mb_state.handle_view is not None:
        bpy.types.SpaceView3D.draw_handler_remove(mb_state.handle_view, 'WINDOW')
        mb_state.handle_view = None
    if mb_state.handle_pixel is not None:
        bpy.types.SpaceView3D.draw_handler_remove(mb_state.handle_pixel, 'WINDOW')
        mb_state.handle_pixel = None
    if mb_on_depsgraph_update in bpy.app.handlers.depsgraph_update_post: bpy.app.handlers.depsgraph_update_post.remove(mb_on_depsgraph_update)
    if mb_on_undo_post in bpy.app.handlers.undo_post: bpy.app.handlers.undo_post.remove(mb_on_undo_post)
    if mb_on_redo_post in bpy.app.handlers.redo_post: bpy.app.handlers.redo_post.remove(mb_on_redo_post)
    if mb_on_save_post in bpy.app.handlers.save_post: bpy.app.handlers.save_post.remove(mb_on_save_post)
    if mb_on_load_post in bpy.app.handlers.load_post: bpy.app.handlers.load_post.remove(mb_on_load_post)
    try: bpy.msgbus.clear_by_owner(mb_state.msgbus_owner)
    except Exception: pass
    mb_state.effects = []

    # 2. UI Text Outline cleanup
    uto_disable_outline()
    if uto__on_load_post in bpy.app.handlers.load_post: bpy.app.handlers.load_post.remove(uto__on_load_post)

    # 3. Halftone cleanup
    if ht_state.handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(ht_state.handle, 'WINDOW')
        ht_state.handle = None
    ht_state.shader = ht_state.batch = None

    # 4. Gushing Chaos cleanup
    if gsh_state.vp_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(gsh_state.vp_handle, "WINDOW")
        gsh_state.vp_handle = None
    if gsh_state.node_handle is not None:
        bpy.types.SpaceNodeEditor.draw_handler_remove(gsh_state.node_handle, "WINDOW")
        gsh_state.node_handle = None
    if gsh_state.border_handle is not None:
        bpy.types.SpaceImageEditor.draw_handler_remove(gsh_state.border_handle, "WINDOW")
        gsh_state.border_handle = None
        
    gsh__free_outline_offscreens()
    gsh_state.batch_cache.clear()
    gsh_state.batch_dirty.clear()
    gsh_state.batch_last_build.clear()
    
    try: gsh_restore_theme()
    except Exception as e: gbfx_state.log_error("unregister restore_theme", e)
    
    if gbfx__on_file_load in getattr(bpy.app.handlers, "load_post", []): bpy.app.handlers.load_post.remove(gbfx__on_file_load)
    if gsh_outline_on_depsgraph_update in bpy.app.handlers.depsgraph_update_post: bpy.app.handlers.depsgraph_update_post.remove(gsh_outline_on_depsgraph_update)
    if gsh_on_frame_change in bpy.app.handlers.frame_change_post: bpy.app.handlers.frame_change_post.remove(gsh_on_frame_change)
    for handler_list, func in (
        (getattr(bpy.app.handlers, "render_init", []), gsh__on_render_init),
        (bpy.app.handlers.render_complete, gsh__on_render_done),
        (bpy.app.handlers.render_cancel, gsh__on_render_done),
        (getattr(bpy.app.handlers, "object_bake_pre", []), gsh__on_render_init),
        (getattr(bpy.app.handlers, "object_bake_complete", []), gsh__on_render_done),
        (getattr(bpy.app.handlers, "object_bake_cancel", []), gsh__on_render_done),
    ):
        if func in handler_list: handler_list.remove(func)

    # 5. Timer Explicit Cleanup
    for timer_func in (mb_fx_tick, mb_selection_poll_tick, gsh_master_tick, _trigger_chaos_rescan):
        if bpy.app.timers.is_registered(timer_func):
            bpy.app.timers.unregister(timer_func)
            
    gsh_state.master_timer_running = False

    # 6. Class Unregister
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
