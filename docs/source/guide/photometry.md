# From flux to magnitude

`render()` returns the irradiance at the observer in W/m², integrated over the
whole solar spectrum: it uses the full solar constant (1361 W/m² by default)
with no filter, detector or atmosphere.

## Broadband magnitude

```python
m = -2.5 * np.log10(flux) - 18.915
```

The zero point is set by the Sun:

$$
-18.915 = m_\odot + 2.5\log_{10}(1361), \qquad m_\odot = -26.75
$$

An eclipsed or occulted satellite gives a flux of exactly zero, so guard the
logarithm:

```python
mag = -2.5 * np.log10(flux) - 18.915 if flux > 0 else np.nan
```

## `AB_mag`

`satlight.tools.AB_mag` expects flux in a 500–600 nm band. Applied directly to
`render()` output it gives a magnitude that is too bright. To use it, render
with the in-band solar irradiance instead of the full solar constant:

```python
r = Renderer(obj_path, geometry=geom, solar_constant=E_in_band)
m_V = AB_mag(float(np.sum(r.render())))
```

The BRDFs have no wavelength dependence, so this assumes each surface reflects
the band the same way it reflects the full spectrum.

## Not modelled

- atmospheric extinction
- seeing and the instrument PSF
- sky background and detector noise
- spectral response of surfaces and detector
