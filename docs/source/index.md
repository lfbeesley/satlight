# satlight

`satlight` is a Python package for producing high-fidelity satellite
lightcurves. It combines orbital mechanics (via Skyfield and SGP4) with
Blender's ray tracer to render physically-based reflected light from
resident space objects (RSOs), for use in space situational awareness (SSA)
research.

```{toctree}
:maxdepth: 2
:caption: Contents

installation
tutorial
api
troubleshooting
```

## Quick links

- {doc}`installation` — getting satlight and its dependencies set up
- {doc}`tutorial` — a full walkthrough: TLE → geometry → render → lightcurve
- {doc}`api` — API reference for `Geometry`, `Renderer`, `brdf`, and `tools`
- {doc}`troubleshooting` — common issues and fixes
