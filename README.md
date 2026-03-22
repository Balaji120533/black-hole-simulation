# Gargantua - Interstellar Black Hole Simulation

This project is a real-time, ray-marched rendering of a black hole. It is written in Python and leverages the **Taichi** library for high-performance, GPU-accelerated graphics.

## Features

- **Gravitational Lensing:** Fully simulates the bending of light around a massive object, warping the background starfield and the accretion disk.
- **Accretion Disk:** A glowing, intense disk of gas orbiting the black hole with relativistic effects like **Doppler Beaming** (one side appears brighter due to its approach velocity).
- **Photon Ring:** The characteristic thin ring of trapped light wrapped closely around the event horizon.
- **Real-Time Rendering:** Utilizes Taichi's GPU backend to ray-march the scene in real-time.
- **Interactive Camera:** You can freely orbit around the black hole, zoom in and out, and toggle automatic rotation.

## Requirements

- Python 3.7+
- Taichi library (`pip install taichi`)

## Installation

1. Clone or download this repository.
2. Install the required dependencies:
   ```bash
   pip install taichi
   ```

## Usage

Run the main simulation script:

```bash
python blackhole.py
```

### Controls
When the simulation window opens, use the following controls:
- **Left Mouse Button (Drag):** Orbit the camera around the black hole.
- **W / S Keys:** Zoom in / out.
- **R Key:** Toggle auto-rotation mode.
- **ESC Key:** Exit the simulation.
