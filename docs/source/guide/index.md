# User guide

The {doc}`tutorial <../tutorial>` gets one light curve working. This guide
explains what each piece is doing and what it assumes, so you can tell a
modelling result from a modelling artefact.

{doc}`geometry`
: Building the observing geometry — from a TLE, from Keplerian elements, or by
  hand. Time handling, pass prediction, and the two independent reasons a
  satellite can be invisible.

{doc}`materials`
: How a BRDF is attached to a mesh component, the seven built-in models, and
  the calling contract your own callable has to satisfy.

{doc}`renderer`
: What `render()` actually does, in order: camera fitting, the pixel-area
  radiometry, shadow rays, bounces, and the observability factor.

{doc}`articulation`
: Moving individual components: Sun-tracking arrays, a repointed antenna.
  How `bpy`, `update()` and `render()` divide the work.

{doc}`earthshine`
: Light reflected off the Earth, treated as an extended source rather than a
  second sun.

{doc}`photometry`
: Flux to magnitude. Read this before comparing anything to a measurement —
  there is a 2.2 magnitude trap in here.

{doc}`performance`
: Where the time goes, and how to parallelise without breaking `bpy`.

{doc}`validation`
: The analytic cases the renderer is checked against, and how accurate it
  measurably is.

```{toctree}
:hidden:

geometry
materials
renderer
articulation
earthshine
photometry
performance
validation
```
