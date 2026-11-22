// Strong, saturated colours per transport mode so the dots stand out on the grey basemap.
export const MODE_COLORS: Record<string, string> = {
  Car: '#e03131', // red
  Bicycle: '#1c7ed6', // blue
  Pedestrian: '#2b8a3e', // green
  All: '#e8590c', // orange
};

export function modeColor(mode: string): string {
  return MODE_COLORS[mode] ?? '#e03131';
}

// OpenFreeMap "Positron": a grey vector basemap that needs no API key.
// (CARTO's raster basemaps now answer every keyless request with an "API KEY REQUIRED" tile.)
export const BASEMAP_STYLE = 'https://tiles.openfreemap.org/styles/positron';
