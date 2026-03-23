import taichi as ti
import math

ti.init(arch=ti.gpu)

RES_X = 1920
RES_Y = 1080
pixels = ti.Vector.field(3, dtype=float, shape=(RES_X, RES_Y))

# ── Camera ──────────────────────────────────────────────────────────────────
cam_pos= ti.Vector.field(3, dtype=float, shape=())
cam_lookat= ti.Vector.field(3, dtype=float, shape=())
cam_up = ti.Vector.field(3, dtype=float, shape=())
fov = 50.0

mouse_prev= ti.Vector.field(2, dtype=float, shape=())
camera_angles= ti.Vector.field(2, dtype=float, shape=())
camera_dist= ti.field(dtype=float, shape=())
auto_rotate= ti.field(dtype=int,   shape=())

# ── Particle system (sparse — disk region only) ──────────────────────────────
NUM_PARTICLES= 600          # very few, only disk-region hot gas
particle_pos= ti.Vector.field(3, dtype=float, shape=NUM_PARTICLES)
particle_vel= ti.Vector.field(3, dtype=float, shape=NUM_PARTICLES)
particle_life= ti.field(dtype=float, shape=NUM_PARTICLES)
particles_screen_field = ti.Vector.field(2, dtype=float, shape=NUM_PARTICLES)

# ── Ray march settings ───────────────────────────────────────────────────────
MAX_STEPS = 600          # more steps for the larger view distance
MAX_DIST  = 45.0
DT        = 0.035        # smaller step → smoother bending

# ── Black hole physics ───
G = 1.0
SCHWARZSCHILD_R= 0.5    # event horizon radius
PHOTON_SPHERE_R= 0.75   # 1.5 × rs  (photons orbit here)
LENSING_STRENGTH= 3.2    # greatly increased so back-side disk bends around

# ── Accretion disk ────
DISK_INNER_R= 0.55     # just outside the event horizon
DISK_OUTER_R= 10.0     # extended horizontally for a massive disk
DISK_DENSITY_M= 3.5      # bright & opaque

# ── Helpers ─

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

# ── Sparse starfield — very few stars (deep space is dark) ───────────────────

@ti.func
def star_field(ray_dir):
    u     = ti.atan2(ray_dir.z, ray_dir.x) / (2.0 * math.pi) + 0.5
    v_ang = ti.asin(ti.max(-1.0, ti.min(1.0, ray_dir.y))) / math.pi + 0.5

    color = ti.Vector([0.0, 0.0, 0.0])

    # Only one scale pass → very sparse field; threshold much tighter
    for k in ti.static(range(2)):
        scale = 300.0 + float(k) * 600.0
        uu = u * scale
        vv = v_ang * scale
        cx = ti.floor(uu)
        cy = ti.floor(vv)
        fx = uu - cx
        fy = vv - cy

        sx= hash_f(cx, cy)
        sy= hash_f(cx + 0.5, cy + 0.5)
        bright= hash_f(cx + 1.1, cy + 2.3)
        temp= hash_f(cx + 4.7, cy + 8.1)

        dx = fx - sx
        dy = fy - sy
        d2 = dx * dx + dy * dy
        # Very tight threshold → only a tiny fraction of cells have a star
        threshold = 0.0004 + bright * 0.0006

        if d2 < threshold:
            intensity = (1.0 - d2 / threshold) * bright * 1.2
            # Stars lean slightly warm/white
            sc = ti.Vector([0.95 + temp * 0.15,
                            0.92 + temp * 0.05,
                            0.88 - temp * 0.15])
            color += sc * intensity

    return color

# ── Gravitational lensing (strong — Interstellar magnitude) ──────────────────

@ti.func
def apply_gravity(pos, dir):
    r_mag     = ti.max(pos.norm(), 0.08)
    force_dir = -pos.normalized()                      # toward center
    # 1/r² but amplified near the photon sphere for that extra warp
    force_mag = (G * LENSING_STRENGTH) / (r_mag * r_mag)
    dir      += force_dir * force_mag * DT
    return dir.normalized()

# ── Camera ray builder ────────────────────────────────────────────────────────

@ti.func
def get_ray_dir(uv, p, l, up):
    f = (l - p).normalized()
    r = f.cross(up).normalized()
    u = r.cross(f).normalized()
    return (f + uv.x * r + uv.y * u).normalized()

# ── Relativistic Doppler beaming (makes approaching side brighter) ────────────

@ti.func
def doppler_boost(pos, ray_dir):
    r_xz  = ti.max(ti.Vector([pos.x, pos.z]).norm(), 0.001)
    tang  = ti.Vector([-pos.z / r_xz, 0.0, pos.x / r_xz])
    beta  = ti.min(ti.sqrt(G / r_xz) * 0.55, 0.6)
    cos_th = tang.dot(-ray_dir)
    return ti.pow(ti.max(0.05, 1.0 + beta * cos_th), 3.5)

# ── Accretion disk density ────────────────────────────────────────────────────

@ti.func
def disk_density(p, t):
    r       = ti.Vector([p.x, p.z]).norm()
    density = 0.0
    if r > DISK_INNER_R and r < DISK_OUTER_R:
        # Widened disk for thicker glowing lensed rings
        vert  = ti.exp(-6.0 * p.y * p.y)
        ang   = ti.atan2(p.z, p.x) + t * 0.25
        # Layered spiral density waves (two frequencies)
        n1    = 0.5 + 0.5 * ti.sin(6.0  * r + 3.0  * ang)
        n2    = 0.5 + 0.5 * ti.sin(18.0 * r + 9.0  * ang + 2.1)
        noise = n1 * 0.65 + n2 * 0.35
        edge  = smoothstep(DISK_INNER_R, DISK_INNER_R + 0.55, r) * \
                (1.0 - smoothstep(DISK_OUTER_R - 1.2, DISK_OUTER_R, r))
        density = vert * noise * edge * DISK_DENSITY_M
    return density

# ── Accretion disk color ─────────────

@ti.func
def disk_color(p, ray_dir):
    r    = ti.Vector([p.x, p.z]).norm()
    frac = ti.max(0.0, ti.min(1.0, (r - DISK_INNER_R) / (DISK_OUTER_R - DISK_INNER_R)))

    # Innermost: white-hot / yellow-white  (>5000 K blackbody)
    # Mid:       brilliant orange-amber   
    # Outer:     deep burnt red
    c_inner = ti.Vector([1.00, 0.90, 0.65])   # hot white-yellow
    c_mid   = ti.Vector([1.00, 0.52, 0.08])   # vivid amber-orange
    c_outer = ti.Vector([0.65, 0.12, 0.01])   # deep red-brown

    col = ti.Vector([0.0, 0.0, 0.0])
    if frac < 0.4:
        col = mix(c_inner, c_mid, frac / 0.4)
    else:
        col = mix(c_mid, c_outer, (frac - 0.4) / 0.6)

    return col * doppler_boost(p, ray_dir)

# ── Camera control kernel ─

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

# ── Init ─────

@ti.kernel
def init_simulation():
    # Slightly elevated angle → see the disk arc wrap over top
    camera_angles[None] = ti.Vector([0.0, 0.22])
    camera_dist[None]   = 14.0
    cam_lookat[None]    = ti.Vector([0.0, 0.0, 0.0])
    cam_up[None]        = ti.Vector([0.0, 1.0, 0.0])
    auto_rotate[None]   = 1
    update_camera()

    # Sparse particles — disk region only, NOT scattered across empty space
    for i in range(NUM_PARTICLES):
        theta            = ti.random() * 2.0 * math.pi
        # Concentrate them in the inner bright ring
        r                = DISK_INNER_R + ti.pow(ti.random(), 1.8) * (DISK_OUTER_R * 0.55 - DISK_INNER_R)
        h                = (ti.random() - 0.5) * 0.04
        particle_pos[i]  = ti.Vector([r * ti.cos(theta), h, r * ti.sin(theta)])
        v_orb            = ti.sqrt(G / r) * 0.9
        tang             = ti.Vector([-ti.sin(theta), 0.0, ti.cos(theta)])
        rad_in           = ti.Vector([-ti.cos(theta), 0.0, -ti.sin(theta)])
        particle_vel[i]  = tang * v_orb + rad_in * 0.04
        particle_life[i] = ti.random()

# ── Particle update ────

@ti.kernel
def update_particles(dt: float):
    for i in range(NUM_PARTICLES):
        p = particle_pos[i]
        v = particle_vel[i]
        p += v * dt
        r  = p.norm()

        if r < DISK_INNER_R * 0.9 or particle_life[i] <= 0.0:
            # Respawn at outer disk edge (not far out in space)
            theta            = ti.random() * 2.0 * math.pi
            r_new            = DISK_OUTER_R * 0.4 + ti.random() * DISK_OUTER_R * 0.15
            particle_pos[i]  = ti.Vector([r_new * ti.cos(theta), (ti.random()-0.5)*0.03, r_new * ti.sin(theta)])
            v_orb            = ti.sqrt(G / r_new) * 0.9
            tang             = ti.Vector([-ti.sin(theta), 0.0, ti.cos(theta)])
            rad_in           = ti.Vector([-ti.cos(theta), 0.0, -ti.sin(theta)])
            particle_vel[i]  = tang * v_orb + rad_in * 0.04
            particle_life[i] = 0.6 + ti.random() * 0.4
        else:
            fdir             = -p.normalized()
            v               += (fdir * (G / (r * r)) - v * 0.08) * dt
            v.y             -= p.y * 6.0 * dt
            particle_vel[i]  = v
            particle_pos[i]  = p
            particle_life[i] -= 0.001

# ── Main render kernel ─

@ti.kernel
def render(t: float):
    aspect = float(RES_X) / float(RES_Y)
    for i, j in pixels:
        uv = ti.Vector([(float(i) / RES_X) * 2.0 - 1.0,
                        (float(j) / RES_Y) * 2.0 - 1.0])
        uv.x *= aspect

        ray_origin    = cam_pos[None]
        ray_dir       = get_ray_dir(uv, cam_pos[None], cam_lookat[None], cam_up[None])
        pos           = ray_origin
        accum_color   = ti.Vector([0.0, 0.0, 0.0])
        transmittance = 1.0
        hit_bh        = False

        for _step in range(MAX_STEPS):
            ray_dir = apply_gravity(pos, ray_dir)
            pos    += ray_dir * DT
            d2c     = pos.norm()

            # Perfect black shadow — event horizon swallows all light
            if d2c < SCHWARZSCHILD_R:
                hit_bh = True
                break

            # ── Accretion disk ────
            dens = disk_density(pos, t)
            if dens > 0.001:
                dc            = disk_color(pos, ray_dir)
                amount        = dens * DT * 1.4         # slightly denser / brighter
                accum_color   += dc * amount * transmittance
                transmittance *= ti.exp(-amount * 0.9)

            if d2c > MAX_DIST or transmittance < 0.01:
                break

        # Sparse starfield (gravitationally lensed by bent ray_dir)
        if not hit_bh and transmittance > 0.01:
            accum_color += star_field(ray_dir) * transmittance

        # ── ACES filmic tone mapping ─────────────────────────────────────────
        cr = accum_color.x
        cg = accum_color.y
        cb = accum_color.z
        cr = cr * (2.51 * cr + 0.03) / (cr * (2.43 * cr + 0.59) + 0.14)
        cg = cg * (2.51 * cg + 0.03) / (cg * (2.43 * cg + 0.59) + 0.14)
        cb = cb * (2.51 * cb + 0.03) / (cb * (2.43 * cb + 0.59) + 0.14)
        pixels[i, j] = ti.Vector([ti.max(0.0, ti.min(1.0, cr)),
                                   ti.max(0.0, ti.min(1.0, cg)),
                                   ti.max(0.0, ti.min(1.0, cb))])

# ── Particle projection ───────────────────────────────────────────────────────

@ti.kernel
def project_particles():
    f        = (cam_lookat[None] - cam_pos[None]).normalized()
    r        = f.cross(cam_up[None]).normalized()
    u        = r.cross(f).normalized()
    asp      = float(RES_X) / float(RES_Y)
    fov_r    = fov * 3.14159265 / 180.0
    tan_hfov = ti.tan(fov_r * 0.5)

    for i in range(NUM_PARTICLES):
        p_rel = particle_pos[i] - cam_pos[None]
        dist  = p_rel.dot(f)
        if dist > 0.1:
            sx = (p_rel.dot(r) / (dist * tan_hfov * asp)) * 0.5 + 0.5
            sy = (p_rel.dot(u) / (dist * tan_hfov))       * 0.5 + 0.5
            particles_screen_field[i] = ti.Vector([sx, sy])
        else:
            particles_screen_field[i] = ti.Vector([-10.0, -10.0])

# ── Main loop ───

window = ti.ui.Window("Black Hole", res=(RES_X, RES_Y))
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

    update_cam_kernel()
    update_particles(0.01)
    render(t)
    canvas.set_image(pixels)
    window.show()
    t += 0.016
