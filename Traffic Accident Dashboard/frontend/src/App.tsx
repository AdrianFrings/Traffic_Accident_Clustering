import { useEffect, useMemo, useState } from 'react';
import { Header } from './components/dashboard/Header';
import { Controls } from './components/dashboard/Controls';
import { RiskMap } from './components/dashboard/RiskMap';
import { HotspotList } from './components/dashboard/HotspotList';
import { ClusterDialog } from './components/dashboard/ClusterDialog';
import { Skeleton } from './components/ui/skeleton';
import { Loader2 } from 'lucide-react';
import { api, type Cluster } from './lib/api';
import { MODE_COLORS, modeColor } from './lib/map';
import { HeroCollage } from './components/dashboard/Collage';

const WRITEUP_URL = 'https://adrianfrings.com/project/traffic-accident-analysis';

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

  const color = modeColor(mode);

  return (
    <div className="min-h-screen">
      <Header />

      <main className="max-w-7xl mx-auto px-4 pt-6 pb-10">
        <section className="hero">
          <div>
            <h1>
              Where traffic accidents <span className="pixel">cluster.</span>
            </h1>
            <p>
              Nine years of injury accidents from the German accident atlas, grouped into hotspots
              with density-based clustering. A few street corners collect a large share of the
              crashes, and those are the places where fixing the street pays off most. Pick a city
              and a road user, then click a hotspot to see what happens there.
            </p>
            <a className="hero-link" href={WRITEUP_URL}>
              Read the write-up ↗
            </a>
          </div>
          <HeroCollage
            city={city || DEFAULT_CITY}
            mode={mode}
            hotspots={clusters.length}
            accidents={totalAccidents}
          />
        </section>

        <div className="mt-6">
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
        </div>

        <div className="mt-5">
          <div>
            <div className="legend mono-caps mb-2.5">
              {(['Car', 'Bicycle', 'Pedestrian', 'All'] as const).map((m) => (
                <span key={m} style={{ opacity: m === mode ? 1 : 0.45 }}>
                  <i style={{ background: MODE_COLORS[m] }} />
                  {m === 'All' ? 'All road users' : m}
                </span>
              ))}
              <span>· bigger dot = more accidents</span>
            </div>
          </div>
        </div>

        <div className="grid lg:grid-cols-3 gap-5">
          <div className="lg:col-span-2 relative h-[440px] lg:h-[660px] map-frame overflow-hidden">
            {/* Map stays mounted so it re-fits to each city; loading is overlaid. */}
            <RiskMap clusters={clusters} mode={mode} onSelect={setSelected} />
            {loading && (
              <div className="absolute inset-0 z-10 grid place-items-center bg-[var(--paper)]/50 backdrop-blur-[2px]">
                <Loader2 className="h-6 w-6 animate-spin" style={{ color }} />
              </div>
            )}
          </div>

          <div className="h-[440px] lg:h-[660px] flex flex-col">
            <h2 className="mono-caps pb-2 mb-2 border-b border-[var(--ink)]">
              Worst hotspots · {city}
            </h2>
            {loading ? (
              <div className="space-y-3 p-1">
                {Array.from({ length: 6 }).map((_, i) => (
                  <Skeleton key={i} className="h-16 w-full rounded-none" />
                ))}
              </div>
            ) : (
              <HotspotList clusters={clusters} onSelect={setSelected} />
            )}
          </div>
        </div>

        <footer className="site-footer mono-caps">
          <span>
            Data: Unfallatlas (Destatis), 2016–2024, dl-de/by-2-0 · DBSCAN, eps 50 m · Map ©{' '}
            <a href="https://openfreemap.org" target="_blank" rel="noopener noreferrer">OpenFreeMap</a>{' '}
            ©{' '}
            <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">
              OpenStreetMap
            </a>{' '}
            contributors
          </span>
          <span>
            Built by <a href="https://adrianfrings.com">Adrian Frings</a>
          </span>
        </footer>
      </main>

      <ClusterDialog cluster={selected} city={city} mode={mode} onClose={() => setSelected(null)} />
    </div>
  );
}

export default function App() {
  return (
    <Dashboard />
  );
}
