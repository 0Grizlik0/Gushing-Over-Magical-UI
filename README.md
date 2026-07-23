# Gushing-Over-Magical-UI

A dark, magical-girl-inspired UI theme for Blender — plus an optional companion add-on that brings the same aesthetic into procedural viewport effects, readability improvements, and a live theme animator.

![Hero screenshot](screenshots/Theme%20MainScreen%20Preview.png)

[![License: GPL v2+](https://img.shields.io/badge/license-GPL--2.0--or--later-blue.svg)](LICENSE)
![Blender](https://img.shields.io/badge/Blender-4.2%20LTS%20--%205.2%20LTS-orange)

---

## What is this?

This repository contains two related, independently-installable projects:

| | What it is | Where to get it |
|---|---|---|
| 🎨 **Theme** — *Gushing Over Blender* | A complete Blender UI theme covering every editor (3D viewport, node editor, sequencer, graph editor, preferences, and more) in a dark-purple palette with magenta, gold, blue, and green accents. | [Blender Extensions Platform], or manually from [`theme/`](theme/) |
| ⚡ **Add-on** — *Gushing Over Blender FX Addon* | An optional companion tool that layers procedural GPU effects, UI readability improvements, and a live theme animator on top of the theme. Fully functional on its own, but tuned to match the theme's palette. | Manually from [`addon/`](addon/) — see [Installation](#installation) below |

**You don't need the add-on to use the theme.** The theme works completely standalone. The add-on is for anyone who wants extra flair on top of it.

---

## Repository structure

```
gushing-over-blender/
├── README.md
├── LICENSE
├── theme/
│   └── gushing_over_blender.xml
├── addon/
│   └── gushing_fx_addon.py
└── screenshots/
    ├── Theme MainScreen Preview.png
    ├── Theme Screen1.png
    ├── Theme Screen2.png
    ├── Theme Screen3.png
    ├── Addon Preview.gif
    ├── Effects FX Addon Preview.gif
    ├── FX Addon Preview.gif
    ├── Halftone Preview.png
    ├── Node Outline Addon Preview.gif
    ├── Outline Addon Preview.gif
    └── Theme Animator Addon Preview.gif
```

---

## Features

### The Theme

A full interface reskin — not just the 3D viewport. Widgets, panels, the node editor, sequencer, graph editor, outliner, preferences, console, and topbar are all covered, so there are no editors left on Blender's default colors.

![Theme screenshot 1](screenshots/Theme%20Screen1.png)
![Theme screenshot 2](screenshots/Theme%20Screen2.png)
![Theme screenshot 3](screenshots/Theme%20Screen3.png)

### The Add-on

The add-on is a small suite of four independent modules — each one can be toggled on or off separately from **Edit > Preferences > Add-ons > Gushing Over Blender FX Addon**.

![Add-on overview](screenshots/Addon%20Preview.gif)

**1. Magia Baiser FX**
Procedural GPU feedback effects that trigger on common actions, so the viewport reacts when you work:
- Delete → *Baiser's Slash*
- Add → *Alice's Toybox*
- Select → *Azure's Ripple*
- Apply Modifier → *Enormita's Grasp*
- Undo → *Alice's Rewind*
- Redo → *Sulfur's Forward*
- Save → *Enormita Salute*
- Error feedback

![Magia Baiser FX demo](screenshots/FX%20Addon%20Preview.gif)
![Magia Baiser FX demo](screenshots/Effects%20FX%20Addon%20Preview.gif)

**2. UI Text Outline**
Adds a configurable outline/halo behind interface text (buttons, labels, panel titles, tooltips) to keep it readable against busy or brightly-colored themes. Choose between Soft Halo, Wide Halo, or Crisp Outline, and dark or light variants.

**3. Halftone Viewport Overlay**
A GPU shader overlay that lays a manga-style screen-tone pattern over the 3D viewport, with adjustable direction (vertical, horizontal, diagonal, vignette), dot density, dot size, rotation, and opacity.

![Halftone overlay](screenshots/Halftone%20Preview.png)

**4. Magical Girls Chaos**
- A **theme color animator** that cycles the active theme's accent colors live.
- **GPU silhouette outlines** around the active object/node in the 3D viewport and node editor.
- An **animated activity border** that appears around the Image Editor while rendering or baking.

![Theme animator demo](screenshots/Theme%20Animator%20Addon%20Preview.gif)
![Silhouette outline — 3D viewport](screenshots/Outline%20Addon%20Preview.gif)
![Silhouette outline — node editor](screenshots/Node%20Outline%20Addon%20Preview.gif)

> ⚠️ **Heads up:** the theme animator and silhouette outlines work by temporarily changing your actual Blender theme colors while active, and restore them when disabled. If you save your user preferences *while the theme animator is running*, the modified colors can be saved as your permanent theme. Turn it off before saving preferences if you want to keep your original theme colors on disk.

---

## Compatibility

- Blender **4.2 LTS** through **5.2 LTS**
- No external Python dependencies required. `numpy` is used if available for a performance boost, but the add-on works without it.
- Windows, macOS, and Linux — no platform-specific code.

---

## Installation

### Theme

**Option A — Blender Extensions Platform (recommended):**
1. In Blender, go to **Edit > Preferences > Get Extensions**.
2. Search for "Gushing Over Blender".
3. Click **Install**.

**Option B — Manual install:**
1. Download [`theme/gushing_over_blender.xml`](theme/) from this repo.
2. In Blender, go to **Edit > Preferences > Themes**.
3. Open the dropdown at the top of the Themes tab and choose **Install Theme...**
4. Select the downloaded `.xml` file.

### Add-on

1. Download the latest `.zip` from the [Releases](../../releases) page (or the [`addon/`](addon/) folder directly).
2. In Blender, go to **Edit > Preferences > Add-ons**.
3. Click the dropdown in the top-right corner and choose **Install from Disk...**, then select the `.zip`.
4. Enable **Gushing Over Blender FX Addon** in the list.
5. Open the 3D Viewport sidebar (press `N`) and find the **Gushing FX** tab to configure it.

---

## Usage

All add-on settings live in two places:
- **3D Viewport > Sidebar (N) > Gushing FX** — quick toggles for each module.
- **Edit > Preferences > Add-ons > Gushing Over Blender FX Addon** — full settings, including the global color palette and per-trigger controls for Magia Baiser FX.

If you're using both the theme and the add-on together, you can sync the add-on's accent colors to the theme with **Load Colors From Theme XML** in the preferences panel.

---

## License

Both the theme and the add-on are released under the **GNU General Public License v2.0 or later** — see [`LICENSE`](LICENSE) for the full text. This matches Blender's own licensing, which add-ons distributed with or for Blender are required to be compatible with.

---

## Credits & Inspiration

Created by **Grizlik**.

Both the theme and the add-on's visual style, color palette, and effect names were inspired by the anime *[Gushing Over Magical Girls](https://en.wikipedia.org/wiki/Gushing_Over_Magical_Girls)* (*Mahō Shōjo ni Akogarete*). This is an unofficial, fan-made project — it is not affiliated with, endorsed by, or sponsored by the creators, publishers, or rights holders of that series, and contains no assets, artwork, or code taken from it.

The add-on was built with the assistance of AI coding tools.

---
