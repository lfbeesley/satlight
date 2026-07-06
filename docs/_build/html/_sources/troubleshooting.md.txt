# Troubleshooting

## `bpy` won't import / Blender not found

- Ensure `bpy` is installed for your Python version: `pip install bpy`
- Some `bpy` versions require running via Blender itself:
  ```bash
  blender --background --python your_script.py
  ```
- Check your Python version matches what the `bpy` wheel supports.

## Model not loading

- Check the file path passed to `Renderer(obj_path=...)` is correct.
- Ensure the model is a valid `.obj` or `.stl` file.
- Try opening the model in Blender manually to verify it isn't corrupted.

## Dark images / zero flux

- Verify `sun_direction` and `observer_direction` are normalised unit vectors.
- Check `solar_constant` is set to a physically reasonable value (e.g. 1361
  W/m² at 1 AU), not left at a placeholder.
- Confirm `distance_to_observer` is in **metres**, not kilometres.
- Check the satellite isn't eclipsed at the requested time
  (`geom.eclipse_type()`).

## Memory issues during rendering

- Reduce `resolution` in the `Renderer` call.
- Simplify the 3D model mesh (fewer polygons).
- Make sure only one `Renderer`/Blender scene is active at a time.
