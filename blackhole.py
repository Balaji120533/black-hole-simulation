import taichi as ti
import math

ti.init(arch=ti.gpu)

# ═══════════════════════════════════════════════════════════════════════════════
#  Resolution & frame buffers
# ═══════════════════════════════════════════════════════════════════════════════
RES_X, RES_Y = 1920, 1080
pixels = ti.Vector.field(3, dtype=float, shape=(RES_X, RES_Y))

# Bloom at quarter resolution for performance
BLOOM_SCALE = 4
BLOOM_RES_X = RES_X // BLOOM_SCALE
BLOOM_RES_Y = RES_Y // BLOOM_SCALE
bloom_buf_a = ti.Vector.field(3, dtype=float, shape=(BLOOM_RES_X, BLOOM_RES_Y))
bloom_buf_b = ti.Vector.field(3, dtype=float, shape=(BLOOM_RES_X, BLOOM_RES_Y))

# ═══════════════════════════════════════════════════════════════════════════════
#  Camera
# ═══════════════════════════════════════════════════════════════════════════════
cam_pos       = ti.Vector.field(3, dtype=float, shape=())
cam_lookat    = ti.Vector.field(3, dtype=float, shape=())
cam_up        = ti.Vector.field(3, dtype=float, shape=())
fov           = 50.0

mouse_prev    = ti.Vector.field(2, dtype=float, shape=())
camera_angles = ti.Vector.field(2, dtype=float, shape=())
camera_dist   = ti.field(dtype=float, shape=())
auto_rotate   = ti.field(dtype=int,   shape=())

# ═══════════════════════════════════════════════════════════════════════════════
#  Sparse particle system (hot gas embers in disk)
# ═══════════════════════════════════════════════════════════════════════════════
NUM_PARTICLES  = 800
particle_pos   = ti.Vector.field(3, dtype=float, shape=NUM_PARTICLES)
particle_vel   = ti.Vector.field(3, dtype=float, shape=NUM_PARTICLES)
particle_life  = ti.field(dtype=float, shape=NUM_PARTICLES)

# ═══════════════════════════════════════════════════════════════════════════════
#  Physics constants
# ═══════════════════════════════════════════════════════════════════════════════
G                = 1.0
M_BH             = 0.25         # Black-hole mass (geometric units, c=G=1)
SCHWARZSCHILD_R  = 2.0 * M_BH   # Rs = 2GM/c²  → 0.5 (event horizon)
PHOTON_SPHERE_R  = 1.5 * SCHWARZSCHILD_R   # 1.5 × Rs = 0.75 — photons orbit
ISCO_R           = 3.0 * SCHWARZSCHILD_R   # 3 × Rs = 1.5 — innermost stable orbit

# ── Accretion disk ────────────────────────────────────────────────────────────
DISK_INNER_R   = 1.5            # At ISCO (physically correct)
DISK_OUTER_R   = 10.0
DISK_DENSITY_M = 3.0

# ── Blackbody temperature (Novikov-Thorne profile) ────────────────────────────
T_MAX          = 11000.0        # Peak disk temperature (Kelvin)
TEMP_PEAK_NORM = 0.4880         # Analytic peak of the N-T profile function

# ── Ray marching ──────────────────────────────────────────────────────────────
MAX_STEPS = 1400        # More steps — geodesics can wind tightly near the ring
MAX_DIST  = 50.0

# ── Bloom post-process ────────────────────────────────────────────────────────
BLOOM_KERNEL    = 10            # Half-width of Gaussian blur kernel
BLOOM_SIGMA     = 18.0          # Gaussian σ²
BLOOM_THRESHOLD = 0.65          # HDR luminance threshold for bloom
BLOOM_INTENSITY = 0.5           # Strength of additive bloom


# ═══════════════════════════════════════════════════════════════════════════════
#  Utility functions
# ═══════════════════════════════════════════════════════════════════════════════

@ti.func
def mix(x, y, a):
    return x * (1.0 - a) + y * a

@ti.func
def smoothstep(edge0, edge1, x):
    t = ti.max(0.0, ti.min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)

@ti.func
def hash_f(x, y):
    v = ti.sin(x * 127.1 + y * 311.7) * 43758.5453
    return v - ti.floor(v)


# ═══════════════════════════════════════════════════════════════════════════════
#  2-D value noise + FBM  (turbulent accretion-disk structure)
# ═══════════════════════════════════════════════════════════════════════════════

@ti.func
def value_noise_2d(x, y):
    ix = ti.floor(x);  iy = ti.floor(y)
    fx = x - ix;       fy = y - iy
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)
    v00 = hash_f(ix,       iy)
    v10 = hash_f(ix + 1.0, iy)
    v01 = hash_f(ix,       iy + 1.0)
    v11 = hash_f(ix + 1.0, iy + 1.0)
    return mix(mix(v00, v10, ux), mix(v01, v11, ux), uy)

@ti.func
def fbm_2d(x, z, t):
    """Fractal Brownian Motion — 3 octaves, slowly animated."""
    ct = ti.cos(t * 0.08);  st = ti.sin(t * 0.08)
    xr = x * ct - z * st
    zr = x * st + z * ct
    val  = 0.0
    amp  = 0.5
    freq = 1.0
    for _ in ti.static(range(3)):
        val  += amp * value_noise_2d(xr * freq, zr * freq)
        amp  *= 0.5
        freq *= 2.17
    return val


# ═══════════════════════════════════════════════════════════════════════════════
#  Sparse starfield (gravitationally lensed)
# ═══════════════════════════════════════════════════════════════════════════════

@ti.func
def star_field(ray_dir):
    u     = ti.atan2(ray_dir.z, ray_dir.x) / (2.0 * math.pi) + 0.5
    v_ang = ti.asin(ti.max(-1.0, ti.min(1.0, ray_dir.y))) / math.pi + 0.5
    color = ti.Vector([0.0, 0.0, 0.0])
    for k in ti.static(range(2)):
        scale = 300.0 + float(k) * 600.0
        uu = u * scale;       vv = v_ang * scale
        cx = ti.floor(uu);    cy = ti.floor(vv)
        fx = uu - cx;         fy = vv - cy
        sx     = hash_f(cx, cy)
        sy     = hash_f(cx + 0.5, cy + 0.5)
        bright = hash_f(cx + 1.1, cy + 2.3)
        temp   = hash_f(cx + 4.7, cy + 8.1)
        dx = fx - sx;  dy = fy - sy
        d2 = dx * dx + dy * dy
        threshold = 0.0004 + bright * 0.0006
        if d2 < threshold:
            intensity = (1.0 - d2 / threshold) * bright * 1.2
            sc = ti.Vector([0.95 + temp * 0.15,
                            0.92 + temp * 0.05,
                            0.88 - temp * 0.15])
            color += sc * intensity
    return color


# ═══════════════════════════════════════════════════════════════════════════════
#  Blackbody color  (Tanner Helland approximation of Planck spectrum)
# ═══════════════════════════════════════════════════════════════════════════════

@ti.func
def blackbody_color(temp_k):
    t = ti.max(10.0, ti.min(400.0, temp_k / 100.0))
    # ── Red channel ──
    r = 0.0
    if t <= 66.0:
        r = 1.0
    else:
        r = 1.292936186 * ti.pow(t - 60.0, -0.1332047592)
    # ── Green channel ──
    g = 0.0
    if t <= 66.0:
        g = 0.390081579 * ti.log(t) - 0.631841444
    else:
        g = 1.129890861 * ti.pow(t - 60.0, -0.0755148492)
    # ── Blue channel ──
    b = 0.0
    if t >= 66.0:
        b = 1.0
    elif t > 19.0:
        b = 0.543206789 * ti.log(t - 10.0) - 1.196254089
    return ti.Vector([ti.max(0.0, ti.min(1.0, r)),
                      ti.max(0.0, ti.min(1.0, g)),
                      ti.max(0.0, ti.min(1.0, b))])


# ═══════════════════════════════════════════════════════════════════════════════
#  Disk temperature — Novikov-Thorne profile
#       T(r) = T_max × (r_isco/r)^(3/4) × [1 − √(r_isco/r)]^(1/4)
#  Peak at r ≈ 1.36 × r_isco ;  zero at r = r_isco  (stress-free boundary)
# ═══════════════════════════════════════════════════════════════════════════════

@ti.func
def disk_temperature(r):
    x   = DISK_INNER_R / ti.max(r, DISK_INNER_R + 0.001)
    raw = ti.pow(x, 0.75) * ti.pow(ti.max(1e-6, 1.0 - ti.sqrt(x)), 0.25)
    return T_MAX * raw / TEMP_PEAK_NORM


# ═══════════════════════════════════════════════════════════════════════════════
#  Gravitational redshift:  z = √(1 − Rs/r)
#  Light from deeper in the potential well is dimmer and redder.
# ═══════════════════════════════════════════════════════════════════════════════

@ti.func
def grav_redshift(r):
    return ti.sqrt(ti.max(0.0, 1.0 - SCHWARZSCHILD_R / ti.max(r, SCHWARZSCHILD_R + 0.01)))


# ═══════════════════════════════════════════════════════════════════════════════
#  Relativistic Doppler beaming  (approaching side brighter)
# ═══════════════════════════════════════════════════════════════════════════════

@ti.func
def doppler_factor(pos, ray_dir):
    # ── Relativistic Doppler factor 𝒟 for matter on a Keplerian orbit ──
    # 𝒟 = 1 / [γ (1 − β·n̂)] ,  with β = v/c the orbital velocity and n̂ the
    # direction from emitter to observer (= −ray_dir along the photon path).
    # Keplerian speed in Schwarzschild:  v_φ = √(M / (r − Rs))  (locally
    # measured), capped below c.  The observed specific intensity then boosts
    # as I_obs = 𝒟⁴ · I_emit  (Liouville's theorem on I_ν/ν³).
    r_xz   = ti.max(ti.Vector([pos.x, pos.z]).norm(), SCHWARZSCHILD_R)
    tang   = ti.Vector([-pos.z / r_xz, 0.0, pos.x / r_xz])
    beta   = ti.min(ti.sqrt(M_BH / ti.max(r_xz - SCHWARZSCHILD_R, 1e-3)), 0.95)
    gamma  = 1.0 / ti.sqrt(ti.max(1e-4, 1.0 - beta * beta))
    cos_th = tang.dot(-ray_dir)
    D      = 1.0 / ti.max(1e-3, gamma * (1.0 - beta * cos_th))
    return D


# ═══════════════════════════════════════════════════════════════════════════════
#  Accretion-disk density  (spiral arms + fractal turbulence)
# ═══════════════════════════════════════════════════════════════════════════════

@ti.func
def disk_density(p, t):
    r = ti.Vector([p.x, p.z]).norm()
    density = 0.0
    if r > DISK_INNER_R and r < DISK_OUTER_R:
        # Vertical Gaussian — thinner near centre, thicker at edge
        h_scale = 0.12 + 0.08 * (r / DISK_OUTER_R)
        vert = ti.exp(-p.y * p.y / (2.0 * h_scale * h_scale))

        ang = ti.atan2(p.z, p.x)

        # Logarithmic spiral arms (two interleaved)
        log_r   = ti.log(ti.max(r, 0.1))
        spiral1 = 0.5 + 0.5 * ti.sin(5.0 * log_r - 2.5 * ang + t * 0.15)
        spiral2 = 0.5 + 0.5 * ti.sin(5.0 * log_r - 2.5 * ang + 3.14159 + t * 0.1)
        spiral  = ti.max(spiral1, spiral2)

        # Fractal turbulence
        noise = fbm_2d(p.x * 1.5, p.z * 1.5, t)

        structure = spiral * 0.55 + noise * 0.45

        # Smooth edge fall-off
        edge = smoothstep(DISK_INNER_R, DISK_INNER_R + 0.8, r) * \
               (1.0 - smoothstep(DISK_OUTER_R - 1.5, DISK_OUTER_R, r))

        # Radial density profile — denser near centre
        radial = 1.0 / (0.3 + r * 0.25)

        density = vert * structure * edge * radial * DISK_DENSITY_M
    return density


# ═══════════════════════════════════════════════════════════════════════════════
#  Accretion-disk color  (blackbody + gravitational redshift + Doppler)
# ═══════════════════════════════════════════════════════════════════════════════

@ti.func
def disk_color(p, ray_dir):
    r = ti.Vector([p.x, p.z]).norm()

    # ── Rest-frame temperature from the Novikov-Thorne profile ──
    temp = disk_temperature(r)

    # ── Combined frequency shift: gravitational × relativistic Doppler ──
    # A photon's frequency is multiplied by g_grav (redshift climbing out of
    # the well) and by the Doppler factor 𝒟.  Since a blackbody stays a
    # blackbody under a frequency shift with T_obs = g · T_emit, we shift the
    # temperature itself — this reddens the receding side and blues the
    # approaching side, exactly as observed.
    g_grav   = grav_redshift(r)
    D        = doppler_factor(p, ray_dir)
    g_total  = g_grav * D
    temp_obs = temp * g_total

    # Blackbody RGB from the *observed* temperature (color shifts correctly)
    col = blackbody_color(temp_obs)

    # ── Stefan-Boltzmann: emitted bolometric intensity ∝ T⁴ ──
    # Observed specific intensity boosts as 𝒟⁴ (relativistic beaming) and the
    # gravitational part folds in through g_total⁴ as well.  We combine the
    # rest-frame T⁴ emission with the g_total⁴ transport factor.
    t_norm    = ti.max(temp, 0.0) / T_MAX
    emit      = ti.pow(t_norm, 4.0)                 # Stefan-Boltzmann ∝ T⁴
    transport = ti.pow(ti.max(g_total, 0.02), 4.0)  # I_obs = g⁴ I_emit
    intensity = emit * transport * 6.0

    return col * intensity


# ═══════════════════════════════════════════════════════════════════════════════
#  Subtle relativistic jet glow  (synchrotron emission along polar axis)
# ═══════════════════════════════════════════════════════════════════════════════

@ti.func
def jet_glow(pos):
    col = ti.Vector([0.0, 0.0, 0.0])
    r = pos.norm()
    if r > 0.6 and r < 12.0:
        r_cyl = ti.sqrt(pos.x * pos.x + pos.z * pos.z)
        abs_y = ti.abs(pos.y)
        cone_r = 0.08 + abs_y * 0.05
        if abs_y > 0.3 and r_cyl < cone_r:
            cross_sec = ti.exp(-r_cyl * r_cyl / (cone_r * cone_r * 0.35))
            falloff   = ti.exp(-abs_y * 0.3)
            intensity = cross_sec * falloff * 0.05
            col = ti.Vector([0.35, 0.50, 1.00]) * intensity
    return col


# ═══════════════════════════════════════════════════════════════════════════════
#  Gravitational lensing  (velocity-Verlet with adaptive step size)
#  Smaller steps near the photon sphere → sharper photon ring
# ═══════════════════════════════════════════════════════════════════════════════

@ti.func
def apply_gravity(pos, dir_vec, dt):
    # ── Exact Schwarzschild null-geodesic deflection ──
    # The photon orbit obeys  d²u/dφ² + u = 3M u²   (u = 1/r).
    # In Cartesian ray-marching form the GR acceleration on the photon is
    #     a = -(3/2) · Rs · (h² / r⁵) · r̂
    # where h = |r × v̂| is the conserved specific angular momentum.
    # This is NOT a Newtonian 1/r² force — the h²/r⁵ term is precisely the
    # post-Newtonian "3Mu²" correction that produces the photon ring and the
    # secondary (over-the-top) image of the disk.
    r_mag = ti.max(pos.norm(), SCHWARZSCHILD_R * 0.5)
    r_hat = pos / r_mag
    h_vec = pos.cross(dir_vec)          # v̂ is already unit length
    h2    = h_vec.dot(h_vec)            # |r × v̂|² = b² (impact-parameter²)
    accel = -1.5 * SCHWARZSCHILD_R * h2 / ti.pow(r_mag, 5.0) * r_hat
    dir_vec += accel * dt
    return dir_vec.normalized()

@ti.func
def adaptive_dt(r):
    """Tiny steps near the photon sphere (sharp ring) → coarse far away.
    The h²/r⁵ deflection is extreme near r≈1.5Rs, so accuracy there is
    what resolves the photon ring and the secondary disk image."""
    DT_MIN = 0.004
    DT_MAX = 0.05
    return DT_MIN + (DT_MAX - DT_MIN) * smoothstep(PHOTON_SPHERE_R, 6.0, r)


# ═══════════════════════════════════════════════════════════════════════════════
#  Camera helpers
# ═══════════════════════════════════════════════════════════════════════════════

@ti.func
def get_ray_dir(uv, p, l, up):
    f = (l - p).normalized()
    r = f.cross(up).normalized()
    u = r.cross(f).normalized()
    return (f + uv.x * r + uv.y * u).normalized()

@ti.func
def update_camera():
    az = camera_angles[None].x
    el = ti.max(-1.4, ti.min(1.4, camera_angles[None].y))
    d  = camera_dist[None]
    cam_pos[None] = ti.Vector([d * ti.cos(el) * ti.sin(az),
                                d * ti.sin(el),
                                d * ti.cos(el) * ti.cos(az)])

@ti.kernel
def update_cam_kernel():
    update_camera()


# ═══════════════════════════════════════════════════════════════════════════════
#  Initialisation
# ═══════════════════════════════════════════════════════════════════════════════

@ti.kernel
def init_simulation():
    camera_angles[None] = ti.Vector([0.0, 0.22])
    camera_dist[None]   = 14.0
    cam_lookat[None]    = ti.Vector([0.0, 0.0, 0.0])
    cam_up[None]        = ti.Vector([0.0, 1.0, 0.0])
    auto_rotate[None]   = 1
    update_camera()

    for i in range(NUM_PARTICLES):
        theta = ti.random() * 2.0 * math.pi
        r     = DISK_INNER_R + ti.pow(ti.random(), 1.8) * \
                (DISK_OUTER_R * 0.55 - DISK_INNER_R)
        h     = (ti.random() - 0.5) * 0.04
        particle_pos[i]  = ti.Vector([r * ti.cos(theta), h,
                                       r * ti.sin(theta)])
        v_orb = ti.sqrt(G / r) * 0.9
        tang  = ti.Vector([-ti.sin(theta), 0.0, ti.cos(theta)])
        rad_i = ti.Vector([-ti.cos(theta), 0.0, -ti.sin(theta)])
        particle_vel[i]  = tang * v_orb + rad_i * 0.04
        particle_life[i] = ti.random()


# ═══════════════════════════════════════════════════════════════════════════════
#  Particle update
# ═══════════════════════════════════════════════════════════════════════════════

@ti.kernel
def update_particles(dt: float):
    for i in range(NUM_PARTICLES):
        p = particle_pos[i]
        v = particle_vel[i]
        p += v * dt
        r = p.norm()

        if r < DISK_INNER_R * 0.9 or particle_life[i] <= 0.0:
            theta = ti.random() * 2.0 * math.pi
            r_new = DISK_OUTER_R * 0.4 + ti.random() * DISK_OUTER_R * 0.15
            particle_pos[i] = ti.Vector([r_new * ti.cos(theta),
                                          (ti.random() - 0.5) * 0.03,
                                          r_new * ti.sin(theta)])
            v_orb = ti.sqrt(G / r_new) * 0.9
            tang  = ti.Vector([-ti.sin(theta), 0.0, ti.cos(theta)])
            rad_i = ti.Vector([-ti.cos(theta), 0.0, -ti.sin(theta)])
            particle_vel[i]  = tang * v_orb + rad_i * 0.04
            particle_life[i] = 0.6 + ti.random() * 0.4
        else:
            fdir = -p.normalized()
            v   += (fdir * (G / (r * r)) - v * 0.08) * dt
            v.y -= p.y * 6.0 * dt
            particle_vel[i]  = v
            particle_pos[i]  = p
            particle_life[i] -= 0.001


# ═══════════════════════════════════════════════════════════════════════════════
#  Main ray-march render kernel  (outputs HDR — no tone mapping yet)
# ═══════════════════════════════════════════════════════════════════════════════

@ti.kernel
def render(t: float):
    aspect = float(RES_X) / float(RES_Y)
    for i, j in pixels:
        uv = ti.Vector([(float(i) / RES_X) * 2.0 - 1.0,
                         (float(j) / RES_Y) * 2.0 - 1.0])
        uv.x *= aspect

        ray_origin    = cam_pos[None]
        ray_dir       = get_ray_dir(uv, cam_pos[None], cam_lookat[None],
                                     cam_up[None])
        pos           = ray_origin
        accum_color   = ti.Vector([0.0, 0.0, 0.0])
        transmittance = 1.0
        hit_bh        = False

        for _step in range(MAX_STEPS):
            d2c = pos.norm()
            dt  = adaptive_dt(d2c)

            ray_dir = apply_gravity(pos, ray_dir, dt)
            pos    += ray_dir * dt
            d2c     = pos.norm()

            # ── Event horizon — perfect black ──
            if d2c < SCHWARZSCHILD_R:
                hit_bh = True
                break

            # ── Accretion-disk volumetric integration ──
            dens = disk_density(pos, t)
            if dens > 0.001:
                dc            = disk_color(pos, ray_dir)
                amount        = dens * dt * 1.4
                accum_color  += dc * amount * transmittance
                transmittance *= ti.exp(-amount * 0.9)

            # ── Jet glow ──
            jg    = jet_glow(pos)
            j_lum = jg.x + jg.y + jg.z
            if j_lum > 1e-5:
                accum_color += jg * dt * transmittance

            if d2c > MAX_DIST or transmittance < 0.01:
                break

        # Gravitationally-lensed starfield
        if not hit_bh and transmittance > 0.01:
            accum_color += star_field(ray_dir) * transmittance

        pixels[i, j] = accum_color          # HDR output


# ═══════════════════════════════════════════════════════════════════════════════
#  Bloom post-process  (extract → blur H → blur V → composite + tone map)
# ═══════════════════════════════════════════════════════════════════════════════

@ti.kernel
def bloom_extract():
    """Down-sample HDR pixels and threshold bright regions."""
    for i, j in bloom_buf_a:
        fi = ti.min(i * BLOOM_SCALE, RES_X - 1)
        fj = ti.min(j * BLOOM_SCALE, RES_Y - 1)
        c  = pixels[fi, fj]
        lum = 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
        if lum > BLOOM_THRESHOLD:
            bloom_buf_a[i, j] = c * ((lum - BLOOM_THRESHOLD) / lum)
        else:
            bloom_buf_a[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def bloom_blur_h():
    """Horizontal Gaussian blur on the bloom buffer."""
    for i, j in bloom_buf_b:
        col = ti.Vector([0.0, 0.0, 0.0])
        tw  = 0.0
        for k in ti.static(range(-BLOOM_KERNEL, BLOOM_KERNEL + 1)):
            ni = i + k
            if ni >= 0 and ni < BLOOM_RES_X:
                w    = ti.exp(-0.5 * float(k * k) / BLOOM_SIGMA)
                col += bloom_buf_a[ni, j] * w
                tw  += w
        bloom_buf_b[i, j] = col / ti.max(tw, 1e-6)

@ti.kernel
def bloom_blur_v():
    """Vertical Gaussian blur — result back into buf_a (ping-pong)."""
    for i, j in bloom_buf_a:
        col = ti.Vector([0.0, 0.0, 0.0])
        tw  = 0.0
        for k in ti.static(range(-BLOOM_KERNEL, BLOOM_KERNEL + 1)):
            nj = j + k
            if nj >= 0 and nj < BLOOM_RES_Y:
                w    = ti.exp(-0.5 * float(k * k) / BLOOM_SIGMA)
                col += bloom_buf_b[i, nj] * w
                tw  += w
        bloom_buf_a[i, j] = col / ti.max(tw, 1e-6)

@ti.func
def bilinear_bloom(u, v):
    """Sample bloom buffer with bilinear interpolation (smooth up-sample)."""
    fx = u * float(BLOOM_RES_X) - 0.5
    fy = v * float(BLOOM_RES_Y) - 0.5
    ix = int(ti.floor(fx));  iy = int(ti.floor(fy))
    wx = fx - ti.floor(fx);  wy = fy - ti.floor(fy)
    ix0 = ti.max(0, ti.min(ix,     BLOOM_RES_X - 1))
    ix1 = ti.max(0, ti.min(ix + 1, BLOOM_RES_X - 1))
    iy0 = ti.max(0, ti.min(iy,     BLOOM_RES_Y - 1))
    iy1 = ti.max(0, ti.min(iy + 1, BLOOM_RES_Y - 1))
    c0 = bloom_buf_a[ix0, iy0] * (1.0 - wx) + bloom_buf_a[ix1, iy0] * wx
    c1 = bloom_buf_a[ix0, iy1] * (1.0 - wx) + bloom_buf_a[ix1, iy1] * wx
    return c0 * (1.0 - wy) + c1 * wy

@ti.kernel
def bloom_composite_and_tonemap():
    """Add bloom to HDR, apply ACES tone mapping + cinematic vignette."""
    for i, j in pixels:
        # Bilinear-up-sampled bloom
        u = (float(i) + 0.5) / float(RES_X)
        v = (float(j) + 0.5) / float(RES_Y)
        bloom_val = bilinear_bloom(u, v)

        # Combine HDR + bloom
        hdr = pixels[i, j] + bloom_val * BLOOM_INTENSITY

        # Subtle cinematic vignette
        vu = float(i) / float(RES_X) - 0.5
        vv = float(j) / float(RES_Y) - 0.5
        d2 = vu * vu + vv * vv
        vignette = 1.0 - smoothstep(0.2, 0.7, d2) * 0.35
        hdr *= vignette

        # ACES filmic tone mapping
        cr = hdr.x;  cg = hdr.y;  cb = hdr.z
        cr = cr * (2.51 * cr + 0.03) / (cr * (2.43 * cr + 0.59) + 0.14)
        cg = cg * (2.51 * cg + 0.03) / (cg * (2.43 * cg + 0.59) + 0.14)
        cb = cb * (2.51 * cb + 0.03) / (cb * (2.43 * cb + 0.59) + 0.14)
        pixels[i, j] = ti.Vector([ti.max(0.0, ti.min(1.0, cr)),
                                   ti.max(0.0, ti.min(1.0, cg)),
                                   ti.max(0.0, ti.min(1.0, cb))])


# ═══════════════════════════════════════════════════════════════════════════════
#  Main loop
# ═══════════════════════════════════════════════════════════════════════════════

window = ti.ui.Window("Black Hole — Realistic Simulation", res=(RES_X, RES_Y))
canvas = window.get_canvas()

init_simulation()

t = 0.0
print("Controls:")
print("  Mouse drag (LMB) — Orbit camera")
print("  W / S             — Zoom in / out")
print("  R                 — Toggle auto-rotation")
print("  ESC               — Exit")

while window.running:
    mouse_curr = window.get_cursor_pos()

    if window.is_pressed(ti.ui.LMB):
        dx = mouse_curr[0] - mouse_prev[None].x
        dy = mouse_curr[1] - mouse_prev[None].y
        camera_angles[None].x -= dx * 5.0
        camera_angles[None].y += dy * 5.0
        auto_rotate[None] = 0

    if window.get_event(ti.ui.PRESS):
        if window.event.key == 'r':
            auto_rotate[None] = 1 - auto_rotate[None]
        elif window.event.key == ti.ui.ESCAPE:
            window.running = False

    if window.is_pressed('w'):
        camera_dist[None] = ti.max(2.0, camera_dist[None] - 0.2)
    if window.is_pressed('s'):
        camera_dist[None] = ti.min(40.0, camera_dist[None] + 0.2)

    if auto_rotate[None] == 1 and not window.is_pressed(ti.ui.LMB):
        camera_angles[None].x += 0.004

    mouse_prev[None] = ti.Vector([mouse_curr[0], mouse_curr[1]])

    # ── Update & render ──
    update_cam_kernel()
    update_particles(0.01)
    render(t)

    # ── Bloom post-process pipeline ──
    bloom_extract()
    bloom_blur_h()
    bloom_blur_v()
    bloom_composite_and_tonemap()

    canvas.set_image(pixels)
    window.show()
    t += 0.016
