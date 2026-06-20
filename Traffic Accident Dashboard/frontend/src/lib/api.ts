// Thin client over the FastAPI backend. In dev, point VITE_API_BASE at the
// local API (http://127.0.0.1:8800); in prod the frontend is served same-origin
// so the default empty base resolves to relative /api paths.
const BASE = import.meta.env.VITE_API_BASE ?? '';

export interface AccidentType {
  code: string;
  label: string;
  count: number;
}

export interface Cluster {
  cluster_id: number;
  lat: number;
  lon: number;
  accident_count: number;
  risk_score: number;
  street_name: string;
  location_name: string;
  accident_types: AccidentType[];
  google_maps_link: string;
}

export interface ClustersResponse {
  city: string;
  mode: string;
  count: number;
  clusters: Cluster[];
}

export interface LatLon {
  lat: number;
  lon: number;
}

export interface ClusterPoints {
  city: string;
  mode: string;
  cluster_id: number;
  cluster: LatLon[];
  noise: LatLon[];
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    throw new Error(`Request failed (${res.status}): ${path}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  cities: () => get<{ cities: string[] }>('/api/cities').then((r) => r.cities),
  modes: (city: string) =>
    get<{ modes: string[] }>(`/api/cities/${encodeURIComponent(city)}/modes`).then((r) => r.modes),
  clusters: (city: string, mode: string) =>
    get<ClustersResponse>(
      `/api/clusters?city=${encodeURIComponent(city)}&mode=${encodeURIComponent(mode)}`,
    ),
  clusterPoints: (clusterId: number, city: string, mode: string) =>
    get<ClusterPoints>(
      `/api/clusters/${clusterId}/points?city=${encodeURIComponent(city)}&mode=${encodeURIComponent(mode)}`,
    ),
};
