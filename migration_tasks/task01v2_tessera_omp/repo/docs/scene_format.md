# The `.tsr` scene format

Plain text, one command per line, `#` starts a comment. The first command must
be the header `tessera-scene 1`. Numbers are whitespace separated.

```
tessera-scene 1
background R G B

camera lookat EX EY EZ  TX TY TZ  UX UY UZ  fov DEG  size W H  [near N] [far F]
camera ortho  EX EY EZ  TX TY TZ  UX UY UZ  scale PX_PER_UNIT  size W H  [near N] [far F]

sh_degree D                       # 0..3, before the first gaussian
gaussian X Y Z  QW QX QY QZ  SX SY SZ  OPACITY  C...   # (D+1)^2 * 3 SH coefficients, DC first, RGB interleaved

mesh NAME
v X Y Z
f A B C                           # 1-based vertex indices, as in OBJ
end

material [grid LEVELS FEATURES LOG2_HASHMAP BASE_RES PER_LEVEL_SCALE]
         [mlp HIDDEN_WIDTH HIDDEN_LAYERS] [seed S]
         [bounds X0 Y0 Z0 X1 Y1 Z1] [tint R G B]
```

## Semantics

* **World space** is right-handed with +y up (the `up` vector of a camera can
  be anything not parallel to its viewing direction).
* **Cameras** follow the OpenCV convention internally: +x right, +y down, +z
  forward. `fov` is the vertical field of view. Orthographic cameras map one
  world unit to `scale` pixels.
* **Gaussians** have a mean, a rotation quaternion `(w, x, y, z)` (normalized on
  use), per-axis standard deviations and an opacity in `[0, 1]`. Radiance is
  `SH(view direction) + 0.5`, clamped at zero, as in 3D Gaussian Splatting.
* **Meshes** are triangle lists. All meshes share one neural **material**: a
  hash-grid encoding of the world position (mapped from `bounds` onto the unit
  cube) followed by a small MLP with a sigmoid output, multiplied by `tint` and
  a two-sided headlight term. Parameters are generated from `seed`, so a scene
  renders identically everywhere.

Splats are composited front to back and stop at the nearest mesh surface,
which then shows through with the remaining transmittance.
