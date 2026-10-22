"""
Análisis de señales en el dominio del tiempo y la frecuencia
Transformada de Fourier con NumPy + Matplotlib
 
Estructura:
  1. Definición de las señales en el tiempo
  2. Cálculo de la Transformada de Fourier
  3. Visualización (magnitud y fase)
  4. Propiedades: linealidad, desplazamiento en el tiempo, escalamiento
"""
import numpy as np
import matplotlib.pyplot as plt
 
# ---------------------------------------------------------------
# 1. DEFINICIÓN DE LAS SEÑALES EN EL DOMINIO DEL TIEMPO
# ---------------------------------------------------------------
fs = 1000                      # frecuencia de muestreo (Hz)
T = 2.0                        # duración (s)
t = np.arange(-T / 2, T / 2, 1 / fs)   # eje de tiempo centrado en 0
N = len(t)
 
def pulso_rect(t, ancho=0.2):
    """Pulso rectangular de amplitud 1 centrado en 0."""
    return np.where(np.abs(t) <= ancho / 2, 1.0, 0.0)
 
def escalon(t):
    """Función escalón unitario u(t)."""
    return np.where(t >= 0, 1.0, 0.0)
 
def senoidal(t, f0=5, A=1.0):
    """Señal senoidal de frecuencia f0 (Hz)."""
    return A * np.sin(2 * np.pi * f0 * t)
 
x_rect = pulso_rect(t)
x_step = escalon(t)
x_sin = senoidal(t)
 
# ---------------------------------------------------------------
# 2. CÁLCULO DE LA TRANSFORMADA DE FOURIER
# ---------------------------------------------------------------
def espectro(x):
    """Devuelve (frecuencias, X(f)) con el cero al centro."""
    X = np.fft.fftshift(np.fft.fft(x)) / fs   # /fs aproxima la TF continua
    f = np.fft.fftshift(np.fft.fftfreq(len(x), d=1 / fs))
    return f, X
 
def fase_limpia(X, umbral=1e-3):
    """Fase con ruido numérico eliminado donde la magnitud es casi cero."""
    fase = np.angle(X)
    fase[np.abs(X) < umbral * np.max(np.abs(X))] = 0
    return fase
 
# ---------------------------------------------------------------
# 3. VISUALIZACIÓN
# ---------------------------------------------------------------
def graficar(x, nombre, xlim_f=50, archivo=None):
    f, X = espectro(x)
    fig, ax = plt.subplots(3, 1, figsize=(9, 8))
    ax[0].plot(t, x)
    ax[0].set(title=f"{nombre} - dominio del tiempo", xlabel="t (s)", ylabel="x(t)")
    ax[1].plot(f, np.abs(X))
    ax[1].set(title="Magnitud |X(f)|", xlabel="f (Hz)", xlim=(-xlim_f, xlim_f))
    ax[2].plot(f, fase_limpia(X))
    ax[2].set(title="Fase ∠X(f)", xlabel="f (Hz)", ylabel="rad", xlim=(-xlim_f, xlim_f))
    for a in ax:
        a.grid(True)
    fig.tight_layout()
    if archivo:
        fig.savefig(archivo, dpi=150)
 
graficar(x_rect, "Pulso rectangular", archivo="01_pulso_rectangular.png")
graficar(x_step, "Escalón unitario", archivo="02_escalon.png")
graficar(x_sin, "Senoidal 5 Hz", archivo="03_senoidal.png")
 
# ---------------------------------------------------------------
# 4. PROPIEDADES DE LA TRANSFORMADA DE FOURIER
# ---------------------------------------------------------------
 
# 4.1 Linealidad: F{a*x1 + b*x2} = a*F{x1} + b*F{x2}
a, b = 2.0, 3.0
f, X_lin = espectro(a * x_rect + b * x_sin)
_, X1 = espectro(x_rect)
_, X2 = espectro(x_sin)
error_lin = np.max(np.abs(X_lin - (a * X1 + b * X2)))
print(f"Linealidad -> error máximo: {error_lin:.2e}")
 
fig, ax = plt.subplots(2, 1, figsize=(9, 6))
ax[0].plot(f, np.abs(X_lin), label="F{a·x1 + b·x2}")
ax[0].plot(f, np.abs(a * X1 + b * X2), "--", label="a·F{x1} + b·F{x2}")
ax[0].set(title="Linealidad", xlim=(-50, 50), xlabel="f (Hz)")
ax[0].legend(); ax[0].grid(True)
 
# 4.2 Desplazamiento en el tiempo: x(t - t0) <-> X(f) e^{-j2πf t0}
# La magnitud no cambia; la fase cambia linealmente con f.
t0 = 0.25
muestras = int(t0 * fs)
x_desp = np.roll(x_rect, muestras)
f, Xd = espectro(x_desp)
_, Xo = espectro(x_rect)
print("Desplazamiento -> diferencia de magnitud máx:",
      f"{np.max(np.abs(np.abs(Xd) - np.abs(Xo))):.2e}")
 
ax[1].plot(f, np.abs(Xo), label="|X| original")
ax[1].plot(f, np.abs(Xd), "--", label="|X| desplazada (igual)")
ax[1].set(title="Desplazamiento en el tiempo: la magnitud no cambia",
          xlim=(-50, 50), xlabel="f (Hz)")
ax[1].legend(); ax[1].grid(True)
fig.tight_layout()
fig.savefig("04_linealidad_desplazamiento.png", dpi=150)
 
fig, ax = plt.subplots(2, 1, figsize=(9, 6))
ax[0].plot(f, fase_limpia(Xo), label="fase original")
ax[0].plot(f, fase_limpia(Xd), label=f"fase desplazada {t0} s")
ax[0].set(title="Desplazamiento: la fase cambia", xlim=(-20, 20), xlabel="f (Hz)")
ax[0].legend(); ax[0].grid(True)
 
# 4.3 Escalamiento: x(a t) <-> (1/|a|) X(f/a)
# Comprimir en el tiempo (a>1) ensancha el espectro.
x_ancho = pulso_rect(t, ancho=0.4)
x_estrecho = pulso_rect(t, ancho=0.2)     # = x_ancho(2t)
f, Xa = espectro(x_ancho)
_, Xe = espectro(x_estrecho)
ax[1].plot(f, np.abs(Xa), label="pulso ancho (0.4 s)")
ax[1].plot(f, np.abs(Xe), label="pulso estrecho (0.2 s)")
ax[1].set(title="Escalamiento: pulso más corto -> espectro más ancho",
          xlim=(-50, 50), xlabel="f (Hz)")
ax[1].legend(); ax[1].grid(True)
fig.tight_layout()
fig.savefig("05_fase_y_escalamiento.png", dpi=150)
 
plt.show()
 