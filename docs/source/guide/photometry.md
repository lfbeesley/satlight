# From flux to magnitude

`render()` returns an irradiance in W/m². Getting from there to a magnitude
you can compare against a measurement is where the modelling assumptions bite,
so this page is worth reading before you plot anything against data.

## The broadband conversion

```{important}
`render()` is **broadband**. It integrates the full solar constant
(1361 W/m² by default) with no spectral response whatsoever — no filter, no
detector QE, no atmospheric transmission.
```

The consistent conversion is therefore the one calibrated against the Sun's
own broadband magnitude:

```python
m = -2.5 * np.log10(flux) - 18.915
```

where the zero point is

$$
-18.915 = m_\odot + 2.5\log_{10}(1361), \qquad m_\odot = -26.75
$$

This is self-consistent: a surface reflecting all the sunlight that reaches it
gives the magnitude the Sun would have at that distance. It is the right
conversion for satlight's output.

## The AB magnitude trap

```{warning}
`satlight.tools.AB_mag` is a **true AB magnitude**, valid only for *in-band*
(500–600 nm) flux. Applied to `render()` output it reads about **2.2 magnitudes
too bright**.
```

The two are not interchangeable, and the failure is silent — the number looks
like a magnitude and sits in a plausible range.

| | `AB_mag(flux)` | `-2.5·log₁₀(flux) − 18.915` |
|---|---|---|
| Expects | flux in a 500–600 nm band | the full broadband flux |
| Zero point | AB system, from $\Delta\nu$ | solar broadband calibration |
| On `render()` output | ~2.2 mag too bright | correct |

`AB_mag` is retained because in-band work is a legitimate use — but it is
in-band work, which means you must also narrow the renderer's input.

## Doing it properly in a band

To get a magnitude in a passband, change the **input** as well as the output:

```{important}
Spectral magnitudes can only be calculated by changing `solar_constant` from
1361 W/m² to the in-band solar irradiance for the bandpass of your choosing.
```

```python
# Roughly the Johnson V band, 500-600 nm
r = Renderer(obj_path, geometry=geom, solar_constant=E_in_band)
flux_band = float(np.sum(r.render()))
m_V = AB_mag(flux_band)
```

What this does **not** do is account for the wavelength dependence of the
surfaces themselves. A BRDF in satlight is grey — one albedo, no spectral
slope — so an in-band render assumes the material reflects the band the same
way it reflects everything. For solar panels, which are strongly coloured,
that assumption is doing real work and you should say so.

```{note}
This is a genuine limitation, not an oversight. Per-wavelength BRDFs would
require spectral reflectance measurements per component, which is the kind of
data that is rarely available for a real spacecraft.
```

## Zero flux is a result, not a failure

```python
flux = float(np.sum(r.render()))
mag = -2.5 * np.log10(flux) - 18.915 if flux > 0 else None
```

`render()` multiplies by `eclipse_type() × is_visible()`, so an eclipsed or
occulted satellite returns **exactly** zero. Guard the logarithm and report
"not observable" rather than a sentinel magnitude — a very faint number here
would be a fabrication.

## Sanity-checking a light curve

Mean magnitude is the diagnostic that catches most modelling errors, because
it depends on the total reflectance rather than on any detail of the geometry.

```{note}
For a large GEO satellite observed from a good site, a mean in the region of
12–13 is the right order. A model producing a mean of 9 is not a bright
satellite — it is a model whose surfaces are reflecting more light than
reaches them. See the energy-conservation warning in
{doc}`materials`.
```

Three checks worth running before trusting a curve:

1. **Is the mean plausible?** Before looking at structure at all.
2. **Does sharpening the peaks change the mean?** It should not, much. If it
   does, the extra peak came from raising the diffuse floor, not from the
   specular lobe.
3. **Is the total hemispherical reflectance ≤ 1?** Integrate the BRDF. This is
   the check that turns "looks right" into "is physical".

## What is deliberately absent

satlight predicts the irradiance arriving at the top of the atmosphere, for a
geometrically perfect observer. Between that and a measurement sit several
things it does not model:

- **Atmospheric extinction** — airmass-dependent, and significant at the low
  elevations where passes begin and end.
- **Seeing and the instrumental PSF** — satlight's pixels are a sampling grid
  on the body, not a focal plane.
- **Sky background and detector noise.**
- **Spectral response** — see above.

For comparison against real photometry, extinction is the one you cannot skip.
