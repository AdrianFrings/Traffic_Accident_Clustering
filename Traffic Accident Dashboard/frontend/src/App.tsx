import { useEffect, useMemo, useState } from 'react';
import { ThemeProvider } from './components/theme-provider';
import { Header } from './components/dashboard/Header';
import { Controls } from './components/dashboard/Controls';
import { RiskMap } from './components/dashboard/RiskMap';
import { HotspotList } from './components/dashboard/HotspotList';
import { ClusterDialog } from './components/dashboard/ClusterDialog';
import { Skeleton } from './components/ui/skeleton';
import { Loader2 } from 'lucide-react';
import { api, type Cluster } from './lib/api';

const DEFAULT_CITY = 'Frankfurt am Main';
const PREFERRED_MODE = 'Car';

function Dashboard() {
  const [cities, setCities] = useState<string[]>([]);
  const [city, setCity] = useState('');
  const [modes, setModes] = useState<string[]>([]);
  const [mode, setMode] = useState('');
  const [clusters, setClusters] = useState<Cluster[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<Cluster | null>(null);

  // Load city list once.
  useEffect(() => {
    api.cities().then((list) => {
      setCities(list);
      setCity(list.includes(DEFAULT_CITY) ? DEFAULT_CITY : list[0] ?? '');
    });
  }, []);

  // Load modes when the city changes; keep current mode if still valid.
  useEffect(() => {
    if (!city) return;
    api.modes(city).then((list) => {
      setModes(list);
      setMode((prev) => (list.includes(prev) ? prev : list.includes(PREFERRED_MODE) ? PREFERRED_MODE : list[0] ?? ''));
    });
  }, [city]);

  // Load clusters when city/mode change.
  useEffect(() => {
    if (!city || !mode) return;
    let active = true;
    setLoading(true);
    api
      .clusters(city, mode)
      .then((r) => active && setClusters(r.clusters))
      .catch(() => active && setClusters([]))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [city, mode]);

  const totalAccidents = useMemo(
    () => clusters.reduce((sum, c) => sum + c.accident_count, 0),
    [clusters],
  );

  return (
    <div className="min-h-screen bg-background text-foreground transition-colors duration-300">
      <Header />

      <main className="max-w-7xl mx-auto px-4 pt-20 pb-10">
        <div className="mb-5">
          <h1 className="text-2xl font-bold tracking-tight">Traffic Accident Risk Dashboard</h1>
          <p className="text-sm text-muted-foreground mt-1 max-w-3xl">
            High-leverage traffic-safety intervention points across Germany, found by
            density-based clustering of government accident data (2016–2024, injury
            accidents). Select a city and road user to explore its worst hotspots.
          </p>
        </div>

        <Controls
          cities={cities}
          modes={modes}
          city={city}
          mode={mode}
          totalClusters={clusters.length}
          totalAccidents={totalAccidents}
          onCityChange={setCity}
          onModeChange={setMode}
        />

        <div className="mt-4 grid lg:grid-cols-3 gap-4">
          <div className="lg:col-span-2 relative h-[420px] lg:h-[640px] rounded-xl border border-border overflow-hidden bg-muted">
            {/* Map stays mounted so it re-fits to each city; loading is overlaid. */}
            <RiskMap clusters={clusters} mode={mode} onSelect={setSelected} />
            {loading && (
              <div className="absolute inset-0 z-10 grid place-items-center bg-background/50 backdrop-blur-[2px]">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            )}
          </div>

          <div className="rounded-xl border border-border bg-card/50 p-3 h-[420px] lg:h-[640px] flex flex-col">
            <h2 className="text-sm font-medium px-1 pb-2">
              High-risk locations
              <span className="text-muted-foreground font-normal"> · worst first</span>
            </h2>
            {loading ? (
              <div className="space-y-2 p-1">
                {Array.from({ length: 6 }).map((_, i) => (
                  <Skeleton key={i} className="h-14 w-full rounded-lg" />
                ))}
              </div>
            ) : (
              <HotspotList clusters={clusters} mode={mode} onSelect={setSelected} />
            )}
          </div>
        </div>

        <p className="text-xs text-muted-foreground mt-6">
          Data source: German Traffic Accident Atlas (Destatis, Unfallatlas), 2016–2024,
          dl-de/by-2-0. Clusters via DBSCAN (eps 50 m).
        </p>
      </main>

      <ClusterDialog cluster={selected} city={city} mode={mode} onClose={() => setSelected(null)} />
    </div>
  );
}

export default function App() {
  return (
    <ThemeProvider defaultTheme="system" storageKey="portfolio-theme">
      <Dashboard />
    </ThemeProvider>
  );
}
