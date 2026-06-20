import type { StyleSpecification } from 'maplibre-gl';

// Semantic colours per transport mode (mirrors the original dashboard).
export const MODE_COLORS: Record<string, string> = {
  Car: '#e11d48', // rose
  Bicycle: '#2563eb', // blue
  Pedestrian: '#16a34a', // green
  All: '#f59e0b', // amber
};

export function modeColor(mode: string): string {
  return MODE_COLORS[mode] ?? '#e11d48';
}

// CartoDB raster basemaps — light (Positron) and dark (Dark Matter) to track
// the site's theme. Raster keeps the dependency surface tiny (no tile server).
const CARTO = (variant: 'light_all' | 'dark_all'): StyleSpecification => ({
  version: 8,
  sources: {
    carto: {
      type: 'raster',
      tiles: [
        `https://a.basemaps.cartocdn.com/${variant}/{z}/{x}/{y}{ratio}.png`,
        `https://b.basemaps.cartocdn.com/${variant}/{z}/{x}/{y}{ratio}.png`,
        `https://c.basemaps.cartocdn.com/${variant}/{z}/{x}/{y}{ratio}.png`,
        `https://d.basemaps.cartocdn.com/${variant}/{z}/{x}/{y}{ratio}.png`,
      ].map((u) => u.replace('{ratio}', '')),
      tileSize: 256,
      attribution:
        '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors © <a href="https://carto.com/attributions">CARTO</a>',
    },
  },
  layers: [{ id: 'carto', type: 'raster', source: 'carto' }],
});

export function basemapStyle(isDark: boolean): StyleSpecification {
  return CARTO(isDark ? 'dark_all' : 'light_all');
}
