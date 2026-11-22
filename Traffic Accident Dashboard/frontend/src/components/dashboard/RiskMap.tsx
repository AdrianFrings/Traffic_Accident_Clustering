import { useEffect, useRef } from 'react';
import maplibregl from 'maplibre-gl';
import type { Cluster } from '../../lib/api';
import { BASEMAP_STYLE, modeColor } from '../../lib/map';

interface Props {
  clusters: Cluster[];
  mode: string;
  onSelect: (cluster: Cluster) => void;
}

const SOURCE_ID = 'clusters';
const LAYER_ID = 'cluster-circles';
const HALO_ID = 'cluster-halo';

function toFeatureCollection(clusters: Cluster[]): GeoJSON.FeatureCollection {
  const max = Math.max(1, ...clusters.map((c) => c.accident_count));
  return {
    type: 'FeatureCollection',
    // Draw the biggest hotspots last so they sit on top.
    features: [...clusters]
      .sort((a, b) => a.accident_count - b.accident_count)
      .map((c) => ({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [c.lon, c.lat] },
        properties: {
          cluster_id: c.cluster_id,
          street: c.street_name,
          count: c.accident_count,
          // 0..1 share of the city's biggest hotspot; dot size grows with its square root.
          size: Math.sqrt(c.accident_count / max),
        },
      })),
  };
}

function fitToClusters(map: maplibregl.Map, clusters: Cluster[]) {
  if (!clusters.length) return;
  const bounds = new maplibregl.LngLatBounds();
  clusters.forEach((c) => bounds.extend([c.lon, c.lat]));
  map.fitBounds(bounds, { padding: 70, maxZoom: 15, duration: 600 });
}

function escapeHtml(s: string) {
  return s.replace(/[&<>"']/g, (ch) => `&#${ch.charCodeAt(0)};`);
}

export function RiskMap({ clusters, mode, onSelect }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const popupRef = useRef<maplibregl.Popup | null>(null);
  // Latest props, read by map event handlers without re-binding them.
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;
  const clustersRef = useRef(clusters);
  clustersRef.current = clusters;
  const modeRef = useRef(mode);
  modeRef.current = mode;

  // Idempotently push the current clusters/mode onto the map and fit the view.
  const syncData = (map: maplibregl.Map) => {
    const fc = toFeatureCollection(clustersRef.current);
    const color = modeColor(modeRef.current);
    const src = map.getSource(SOURCE_ID) as maplibregl.GeoJSONSource | undefined;
    if (!src) {
      map.addSource(SOURCE_ID, { type: 'geojson', data: fc });
      // Soft coloured halo so hotspots read from a distance.
      map.addLayer({
        id: HALO_ID,
        type: 'circle',
        source: SOURCE_ID,
        paint: {
          'circle-radius': ['+', 16, ['*', 18, ['get', 'size']]],
          'circle-color': color,
          'circle-opacity': 0.18,
          'circle-blur': 0.4,
        },
      });
      map.addLayer({
        id: LAYER_ID,
        type: 'circle',
        source: SOURCE_ID,
        paint: {
          'circle-radius': ['+', 9, ['*', 12, ['get', 'size']]],
          'circle-color': color,
          'circle-opacity': 0.92,
          'circle-stroke-width': 2.5,
          'circle-stroke-color': '#ffffff',
        },
      });
    } else {
      src.setData(fc);
      map.setPaintProperty(LAYER_ID, 'circle-color', color);
      map.setPaintProperty(HALO_ID, 'circle-color', color);
    }
    fitToClusters(map, clustersRef.current);
  };

  // Initialise the map once.
  useEffect(() => {
    if (!containerRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: BASEMAP_STYLE,
      center: [10.45, 51.16], // Germany
      zoom: 5,
      attributionControl: { compact: true },
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');
    mapRef.current = map;
    popupRef.current = new maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 16 });
    if (import.meta.env.DEV) (window as unknown as { __riskmap?: maplibregl.Map }).__riskmap = map;

    const resizeObserver = new ResizeObserver(() => map.resize());
    resizeObserver.observe(containerRef.current);

    // Reads the latest refs, so clusters that arrive before style load still apply.
    map.on('load', () => syncData(map));

    map.on('mouseenter', LAYER_ID, (e) => {
      map.getCanvas().style.cursor = 'pointer';
      const f = e.features?.[0];
      if (!f || !popupRef.current) return;
      const { street, count } = f.properties as { street: string; count: number };
      const [lon, lat] = (f.geometry as GeoJSON.Point).coordinates;
      popupRef.current
        .setLngLat([lon, lat])
        .setHTML(`<b>${escapeHtml(street)}</b><br/>${count} accidents · click for details`)
        .addTo(map);
    });
    map.on('mouseleave', LAYER_ID, () => {
      map.getCanvas().style.cursor = '';
      popupRef.current?.remove();
    });
    map.on('click', LAYER_ID, (e) => {
      const id = (e.features?.[0]?.properties as { cluster_id?: number })?.cluster_id;
      const hit = clustersRef.current.find((c) => c.cluster_id === id);
      if (hit) {
        popupRef.current?.remove();
        onSelectRef.current(hit);
      }
    });

    return () => {
      resizeObserver.disconnect();
      map.remove();
      mapRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Apply data/mode changes. If the style isn't ready yet, the 'load' handler picks up the latest refs.
  useEffect(() => {
    const map = mapRef.current;
    if (map && map.isStyleLoaded()) syncData(map);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clusters, mode]);

  return <div ref={containerRef} className="h-full w-full" />;
}
