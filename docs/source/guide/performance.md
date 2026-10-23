# Performance

Most of the run time is spent casting rays, one per pixel where the
spacecraft appears. These options add more:

| Option | Extra cost per lit pixel |
|---|---|
| `add_shadows=True` | 1 shadow ray |
| `n_bounce=2` | `bounce_samples` rays (8 by default) |
| `n_bounce=3` | `bounce_samples²` rays (64 by default) |
| `add_earthshine=True` | one BRDF evaluation per Earth patch (88 by default) |
| `earthshine_shadows=True` | one shadow ray per Earth patch |

Run time also scales roughly with the number of pixels, so `resolution` is
the main setting to reduce. Check that the answer doesn't change:

```python
for n in (200, 300, 500):
    r = Renderer(obj_path, geometry=geom, resolution=(n, n), brdf=brdf)
    r.initialise_scene()
    print(n, float(np.sum(r.render())))
```

## Skip epochs that can't be seen

`render()` returns zeros immediately when the satellite is eclipsed or below
the horizon. Checking a whole pass first with a multi-epoch `Geometry` (see
{doc}`geometry`) avoids building geometry for frames you don't need.

## Running epochs in parallel

Each epoch is independent, so a light curve can be split across processes.
Threads won't work, because Blender has one shared scene per process.

- Use the `spawn` start method.
- Import satlight (and so `bpy`) inside the worker function, not in the parent.
- Create one `Renderer` per worker and reuse it with `update()`.
- Set `OMP_NUM_THREADS=1` (and the other BLAS thread variables) before
  importing numpy, so the workers don't compete for cores.

```python
import multiprocessing as mp

def render_slice(epochs):
    import numpy as np
    from satlight import Geometry, Renderer
    out, r = [], None
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
        r.update(g)
        out.append(float(np.sum(r.render())))
    return out

if __name__ == "__main__":
    with mp.get_context("spawn").Pool(4) as pool:
        results = pool.map(render_slice, slices)
```
