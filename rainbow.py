import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
from matplotlib.legend_handler import HandlerPatch
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# Paramètres
sun_alt = 20.0   # Soleil 20° au-dessus de l'horizon
antisolar_alt = -sun_alt
antisolar_az = 0.0

# Chargement du CSV
df = pd.read_csv("/Users/matteo_crespinjouan/Downloads/Filtered_Refractive_Index_Data.csv")
wls = df["Wavelength (μm)"].values
ns = df["Refractive Index (n)"].values

def rainbow_angle(n):
    val = (n**2 - 1)/3.0
    if val <= 0:
        return None
    alpha = np.arccos(np.sqrt(val))
    theta = np.arcsin(np.sin(alpha)/n)
    D_min = np.pi + 2*alpha -4*theta
    angle_deg = 180.0 - (D_min * 180.0/np.pi)
    return angle_deg

r_angles = [rainbow_angle(n) for n in ns]

AltC = np.radians(antisolar_alt)
AzC = np.radians(antisolar_az)
sinC = np.sin(AltC)
cosC = np.cos(AltC)

def f_r(Alt_rad, r_deg, Az_deg):
    r = np.radians(r_deg)
    Az = np.radians(Az_deg)
    return np.cos(r) - (sinC*np.sin(Alt_rad) + cosC*np.cos(Alt_rad)*np.cos(Az))

az_range = np.linspace(-60, 60, 200)

# Définition des plages
UV_limit = 0.38
IR_limit = 0.75

def wavelength_to_color_visible(wl):
    wl_nm = wl*1000.0
    if wl_nm < 380: wl_nm = 380
    if wl_nm > 750: wl_nm = 750

    def segment(x, x1, x2, c1, c2):
        return c1 + (c2-c1)*((x - x1)/(x2 - x1))

    R=0.0; G=0.0; B=0.0
    if 380 <= wl_nm < 440:
        R = segment(wl_nm,380,440,0.6,0.0)
        G = 0.0
        B = 1.0
    elif 440 <= wl_nm < 490:
        R = 0.0
        G = segment(wl_nm,440,490,0.0,1.0)
        B = 1.0
    elif 490 <= wl_nm < 530:
        R = 0.0
        G = 1.0
        B = segment(wl_nm,490,530,1.0,0.0)
    elif 530 <= wl_nm < 580:
        R = segment(wl_nm,530,580,0.0,1.0)
        G = 1.0
        B = 0.0
    elif 580 <= wl_nm < 620:
        R = 1.0
        G = segment(wl_nm,580,620,1.0,0.0)
        B = 0.0
    else:
        R = 1.0
        G = 0.0
        B = 0.0
    return (R,G,B)

def wavelength_to_color(wl):
    if wl < UV_limit:
        return (1.0,0.0,1.0)
    elif wl > IR_limit:
        return (0.5,0.0,0.0)
    else:
        return wavelength_to_color_visible(wl)

uv_ir_data = []
visible_data = []

for wl, r in zip(wls, r_angles):
    if r is not None:
        if wl < UV_limit or wl > IR_limit:
            uv_ir_data.append((wl,r))
        else:
            visible_data.append((wl,r))

fig, ax = plt.subplots(figsize=(10,6))
ax.set_aspect('equal', 'box')

def compute_arc_points(r, az_range):
    az_points = []
    alt_points = []
    for Az_deg in az_range:
        low, high = 0.0, 90.0
        found = False
        for _ in range(50):
            mid = 0.5*(low+high)
            val_mid = f_r(np.radians(mid), r, Az_deg)
            if abs(val_mid)<1e-3:
                found = True
                alt_sol = mid
                break
            val_low = f_r(np.radians(low), r, Az_deg)
            if val_mid*val_low <0:
                high = mid
            else:
                low = mid
        if found and alt_sol >= 0:
            az_points.append(Az_deg)
            alt_points.append(alt_sol)
    return az_points, alt_points

# On trace d'abord UV et IR
for wl, r in uv_ir_data:
    az_points, alt_points = compute_arc_points(r, az_range)
    if len(az_points)>0:
        color = wavelength_to_color(wl)
        ax.plot(az_points, alt_points, color=color, linewidth=1)

# Ensuite on trace le visible
for wl, r in visible_data:
    az_points, alt_points = compute_arc_points(r, az_range)
    if len(az_points)>0:
        color = wavelength_to_color(wl)
        ax.plot(az_points, alt_points, color=color, linewidth=2)

ax.axhline(y=0, color='black', linewidth=1)
ax.set_xlim(-60,60)
ax.set_ylim(0,50)
ax.set_xlabel("Azimut (°)")
ax.set_ylabel("Altitude (°)")
ax.set_title("Arc-en-ciel (de 0.2 à 7 microns, pas de 0.025)\nSoleil 20° derrière, arc visible au-dessus de l'horizon")
plt.grid(True)

# Création d'un dégradé pour la légende visible 
# On crée un colormap continue du spectre visible
cmap = mcolors.LinearSegmentedColormap.from_list('visible', [
    (1.0,0.0,1.0), # environ 380 nm
    (0.0,0.0,1.0), # bleu
    (0.0,1.0,1.0), # cyan
    (0.0,1.0,0.0), # vert
    (1.0,1.0,0.0), # jaune
    (1.0,0.0,0.0), # rouge
    (0.5,0.0,0.0)  # vers 750 nm
])

class HandlerGradient(HandlerPatch):
    def __init__(self, cmap, **kw):
        self.cmap = cmap
        super().__init__(**kw)

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # On va créer une série de petits rectangles colorés pour simuler un dégradé
        n = 50
        polys = []
        colors = []
        for i in range(n):
            y1 = ydescent + (i/n)*height
            y2 = ydescent + ((i+1)/n)*height
            polys.append([(xdescent,y1), (xdescent+width,y1),
                          (xdescent+width,y2), (xdescent,y2)])
            color_val = i/(n-1)
            colors.append(self.cmap(color_val))
        poly = PolyCollection(polys, facecolors=colors, edgecolors='none', transform=trans)
        return [poly]

# On crée un handle factice pour la légende du visible
visible_handle = mpatches.Rectangle((0,0),1,1) 

legend_elements = [
    Line2D([0], [0], color=(1.0,0.0,1.0), lw=2, label='UV (<0.38 µm)'),
    visible_handle,
    Line2D([0], [0], color=(0.5,0.0,0.0), lw=2, label='IR (>0.75 µm)')
]

ax.legend(
    handles=legend_elements,
    labels=['UV (<0.38 µm)', 'Visible (0.38–0.75 µm)', 'IR (>0.75 µm)'],
    handler_map={visible_handle: HandlerGradient(cmap=cmap)},
    loc='upper right'
)

plt.show()