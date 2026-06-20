import { useEffect, useRef } from 'react';
import maplibregl from 'maplibre-gl';
import type { ClusterPoints } from '../../lib/api';
import { basemapStyle } from '../../lib/map';
import { useIsDark } from '../../lib/useIsDark';

interface Props {
  points: ClusterPoints;
  color: string;
}

function fc(coords: { lat: number; lon: number }[]): GeoJSON.FeatureCollection {
  return {
    type: 'FeatureCollection',
    features: coords.map((p) => ({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [p.lon, p.lat] },
      properties: {},
    })),
  };
}

export function ClusterZoomMap({ points, color }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const isDark = useIsDark();

  useEffect(() => {
    if (!containerRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: basemapStyle(isDark),
      center: [points.cluster[0]?.lon ?? 10, points.cluster[0]?.lat ?? 51],
      zoom: 16,
      attributionControl: { compact: true },
    });
    mapRef.current = map;

    const resizeObserver = new ResizeObserver(() => map.resize());
    resizeObserver.observe(containerRef.current);

    map.on('load', () => {
      map.addSource('noise', { type: 'geojson', data: fc(points.noise) });
      map.addLayer({
        id: 'noise',
        type: 'circle',
        source: 'noise',
        paint: { 'circle-radius': 4, 'circle-color': '#9ca3af', 'circle-opacity': 0.35 },
      });
      map.addSource('pts', { type: 'geojson', data: fc(points.cluster) });
      map.addLayer({
        id: 'pts',
        type: 'circle',
        source: 'pts',
        paint: {
          'circle-radius': 7,
          'circle-color': color,
          'circle-opacity': 0.7,
          'circle-stroke-width': 1,
          'circle-stroke-color': '#ffffff',
        },
      });

      const bounds = new maplibregl.LngLatBounds();
      points.cluster.forEach((p) => bounds.extend([p.lon, p.lat]));
      if (!bounds.isEmpty()) map.fitBounds(bounds, { padding: 70, maxZoom: 18, duration: 0 });
    });

    return () => {
      resizeObserver.disconnect();
      map.remove();
      mapRef.current = null;
    };
    // Points are stable for the dialog's lifetime; deliberately mount once.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return <div ref={containerRef} className="h-full w-full rounded-lg overflow-hidden" />;
}
