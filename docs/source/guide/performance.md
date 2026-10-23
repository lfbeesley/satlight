# Performance

A single render is a few hundred milliseconds. A night of a LEO pass at 1 Hz
is thousands of renders, and at that point it is worth knowing where the time
actually goes.

## The cost model

Per epoch, the work divides roughly as:

| Stage | Share of a typical frame |
|---|---|
| **Ray casting** (`scene.ray_cast`) | ~54% |
| BRDF evaluation | depends entirely on the model |
| Orbit propagation (Skyfield/SGP4) | tens of ms, independent of resolution |
| Scene update (`view_layer.update`, depsgraph) | small, but once per update |

```{note}
The ray cast dominates, and it scales with the number of **lit pixels**, not
with the frame area — the renderer builds its pixel mask from the per-object
bounding boxes and only casts where the body projects.
```

Multipliers on the ray-cast count, worst first:

| Option | Extra rays per lit pixel |
|---|---|
| `earthshine_shadows=True` | one per patch (88 at `n_rings=4`) |
| `n_bounce=3` | `bounce_samples²` — 64× at the default |
| `add_earthshine=True` | 88 BRDF evaluations (no extra *casts*) |
| `n_bounce=2` | `bounce_samples` — 8× |
| `add_shadows=True` | 1 |

So the cheap wins, in order: drop `earthshine_shadows`, keep `n_bounce` at 1
or 2, and reduce `resolution` before anything else.

## Resolution is the first dial

Cost is roughly linear in pixel count, so 500² → 300² is a 2.8× saving. The
question is whether the answer moves.

```python
for n in (200, 300, 500, 700):
    r = Renderer(obj, geometry=geom, resolution=(n, n), brdf=brdf)
    r.initialise_scene()
    print(n, float(np.sum(r.render())))
```

Convergence is a property of the geometry. A flat plate converges almost
immediately; a narrow specular lobe may not converge at 500² at all. See
{doc}`validation`.

## Pin the BLAS threads

This one is counter-intuitive and worth doing before anything else.

```python
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
# then import numpy / skyfield / satlight
```

```{important}
Set these **before** importing numpy. They are read at import time.
```

Skyfield's numpy calls will otherwise spread across every core — in one
measurement, 6.13 cores' worth of work for a single-epoch propagation. That
looks like the code is already parallel, and it defeats any attempt to run
several epochs at once, because each worker then fights the others for the
same cores.

```{note}
Blender's ray casting is **not** multithreaded here. `scene.ray_cast` is
called once per pixel from Python, so the loop is serial and holds the GIL
between calls. An earlier claim in satlight's own notes that the depsgraph was
doing this work in parallel was wrong; the parallelism being observed was
numpy's.
```

## Parallelising over epochs

A light curve is embarrassingly parallel in time: each epoch is an independent
geometry and an independent render. That is the axis to parallelise on.

**Threads do not work.** `bpy` holds a single global Blender application
state, and the per-pixel ray cast loop holds the GIL. Two threads rendering
two epochs share one scene and will corrupt each other.

**Processes do**, with one trap:

```{warning}
A child process cannot import `bpy` if the parent already has. You get
`ModuleNotFoundError: No module named '_bpy'`, because `bpy`'s extension
module is not re-initialisable in a spawned interpreter.
```

The pattern that works:

1. Use the `spawn` start method.
2. Do **not** import `satlight` (and therefore `bpy`) in the parent. Import it
   inside the worker function.
3. Give each worker a slice of epochs; have it build its own `Geometry` and
   its own `Renderer` and return fluxes, not Blender objects.

```python
import multiprocessing as mp

def render_slice(args):
    import numpy as np                       # imports happen IN the child
    from satlight import Geometry, Renderer
    epochs, obj_path, brdf = args
    out = []
    r = None
    for when in epochs:
        g = Geometry()
        g.create_observer(observer_lat=28.7622, observer_lon=-17.8897,
                          observer_alt_m=2396)
        g.create_satellite(line1, line2)
        g.set_time(when.year, when.month, when.day,
                   when.hour, when.minute, when.second)
        if r is None:
            r = Renderer(obj_path, geometry=g, brdf=brdf)
            r.initialise_scene()
        else:
            r.update(g)
        out.append(float(np.sum(r.render())))
    return out

if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    with ctx.Pool(4) as pool:
        results = pool.map(render_slice, slices)
```

Note the renderer is built **once per worker** and reused via `update()`.
Re-importing the mesh per epoch is wasteful and is usually the reason a
parallel version is no faster than a serial one.

```{note}
With BLAS pinned to one thread per worker, this measured about **2.8×** on a
machine where the unpinned version managed 0.6× — i.e. slower than serial.
The pinning is not a refinement; it is the difference between a speed-up and a
slow-down.
```

## Skip the epochs you do not need

The cheapest render is the one you do not do.

```python
geom.set_time(2024, 1, 15, 19, range(0, 60), 0)    # vectorised screening
```

A vectorised `Geometry` evaluates visibility and eclipse for many epochs at
once, with no Blender involved. Screen first, then render only the epochs
where `observable_fraction() > 0`.

`render()` already short-circuits to a zero array when
`observable_fraction == 0.0`, which for a LEO time series skips most epochs
outright — but the `Geometry` construction and propagation still cost
something, so screening in bulk is faster still.

## What is not in the package

```{important}
There is no `parallel=` or `device='cuda'` option in satlight today. The
patterns above are what you can do from outside it.
```

A GPU path is not a drop-in change. The reason the renderer casts one ray per
pixel from Python — rather than handing Blender a batch, or using a `BVHTree`
directly — is that it needs to know **which named component** each ray hit, in
order to pick that component's BRDF and to attribute a peak afterwards. That
per-hit attribution is a requirement, not an implementation detail, and it is
what a batched or GPU-resident approach would have to preserve.
