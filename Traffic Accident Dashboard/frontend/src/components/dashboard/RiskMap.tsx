import { useEffect, useRef } from 'react';
import maplibregl from 'maplibre-gl';
import type { Cluster } from '../../lib/api';
import { basemapStyle, modeColor } from '../../lib/map';
import { useIsDark } from '../../lib/useIsDark';

interface Props {
  clusters: Cluster[];
  mode: string;
  onSelect: (cluster: Cluster) => void;
}

const SOURCE_ID = 'clusters';
const LAYER_ID = 'cluster-circles';

function toFeatureCollection(clusters: Cluster[]): GeoJSON.FeatureCollection {
  return {
    type: 'FeatureCollection',
    features: clusters.map((c) => ({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [c.lon, c.lat] },
      properties: { cluster_id: c.cluster_id, street: c.street_name, count: c.accident_count },
    })),
  };
}

function fitToClusters(map: maplibregl.Map, clusters: Cluster[]) {
  if (!clusters.length) return;
  const bounds = new maplibregl.LngLatBounds();
  clusters.forEach((c) => bounds.extend([c.lon, c.lat]));
  map.fitBounds(bounds, { padding: 60, maxZoom: 15, duration: 600 });
}

export function RiskMap({ clusters, mode, onSelect }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const popupRef = useRef<maplibregl.Popup | null>(null);
  const isDark = useIsDark();
  // Latest props, read by map event handlers without re-binding them.
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;
  const clustersRef = useRef(clusters);
  clustersRef.current = clusters;
  const modeRef = useRef(mode);
  modeRef.current = mode;

  // Idempotently push the current clusters/mode onto the map and fit the view.
  // Used both when the style first loads and on every data/mode/theme change,
  // so the fit never depends on a transient "idle" event that may not re-fire.
  const syncData = (map: maplibregl.Map) => {
    const fc = toFeatureCollection(clustersRef.current);
    const src = map.getSource(SOURCE_ID) as maplibregl.GeoJSONSource | undefined;
    if (!src) {
      map.addSource(SOURCE_ID, { type: 'geojson', data: fc });
      map.addLayer({
        id: LAYER_ID,
        type: 'circle',
        source: SOURCE_ID,
        paint: {
          'circle-radius': ['interpolate', ['linear'], ['get', 'count'], 1, 6, 100, 18],
          'circle-color': modeColor(modeRef.current),
          'circle-opacity': 0.7,
          'circle-stroke-width': 1.5,
          'circle-stroke-color': '#ffffff',
        },
      });
    } else {
      src.setData(fc);
      map.setPaintProperty(LAYER_ID, 'circle-color', modeColor(modeRef.current));
    }
    fitToClusters(map, clustersRef.current);
  };

  // Initialise the map once.
  useEffect(() => {
    if (!containerRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: basemapStyle(isDark),
      center: [10.45, 51.16], // Germany
      zoom: 5,
      attributionControl: { compact: true },
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');
    mapRef.current = map;
    popupRef.current = new maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 12 });
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
      popupRef.current.setLngLat([lon, lat]).setHTML(`<b>${street}</b><br/>${count} accidents`).addTo(map);
    });
    map.on('mouseleave', LAYER_ID, () => {
      map.getCanvas().style.cursor = '';
      popupRef.current?.remove();
    });
    map.on('click', LAYER_ID, (e) => {
      const id = (e.features?.[0]?.properties as { cluster_id?: number })?.cluster_id;
      const hit = clustersRef.current.find((c) => c.cluster_id === id);
      if (hit) onSelectRef.current(hit);
    });

    return () => {
      resizeObserver.disconnect();
      map.remove();
      mapRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Apply data/mode changes. If the style isn't ready yet, the 'load' handler
  // above will pick up the latest refs.
  useEffect(() => {
    const map = mapRef.current;
    if (map && map.isStyleLoaded()) syncData(map);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clusters, mode]);

  // Swap basemap on theme change. Skip the first run (map already has the right
  // style); on later changes re-apply data once the new style is in.
  const didInitTheme = useRef(false);
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (!didInitTheme.current) {
      didInitTheme.current = true;
      return;
    }
    map.setStyle(basemapStyle(isDark));
    map.once('styledata', () => syncData(map));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isDark]);

  return <div ref={containerRef} className="h-full w-full" />;
}
