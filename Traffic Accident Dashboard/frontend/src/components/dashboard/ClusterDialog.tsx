import { useEffect, useState } from 'react';
import { ExternalLink, MapPin } from 'lucide-react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '../ui/dialog';
import { Button } from '../ui/button';
import { Skeleton } from '../ui/skeleton';
import { AccidentTypeChart } from './AccidentTypeChart';
import { ClusterZoomMap } from './ClusterZoomMap';
import { api, type Cluster, type ClusterPoints } from '../../lib/api';
import { modeColor } from '../../lib/map';

interface Props {
  cluster: Cluster | null;
  city: string;
  mode: string;
  onClose: () => void;
}

export function ClusterDialog({ cluster, city, mode, onClose }: Props) {
  const [points, setPoints] = useState<ClusterPoints | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!cluster) return;
    let active = true;
    setPoints(null);
    setError(false);
    setLoading(true);
    api
      .clusterPoints(cluster.cluster_id, city, mode)
      .then((p) => active && setPoints(p))
      .catch(() => active && setError(true))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [cluster, city, mode]);

  const color = modeColor(mode);

  return (
    <Dialog open={!!cluster} onOpenChange={(open: boolean) => !open && onClose()}>
      <DialogContent className="w-[95vw] sm:max-w-5xl max-h-[90vh] overflow-y-auto bg-[var(--paper)] border-[var(--rule)] rounded-none">
        {cluster && (
          <>
            <DialogHeader className="mb-2">
              <p className="mono-caps">Hotspot detail · {mode === 'All' ? 'all road users' : mode}</p>
              <DialogTitle className="flex items-center gap-2 text-3xl font-semibold tracking-[-0.04em]">
                <MapPin className="h-6 w-6 shrink-0" style={{ color }} />
                {cluster.street_name}
              </DialogTitle>
              {cluster.location_name && cluster.location_name !== cluster.street_name && (
                <p className="mono-caps pl-8">{cluster.location_name}</p>
              )}
            </DialogHeader>

            <div className="grid md:grid-cols-2 gap-8">
              <div className="flex flex-col">
                <div className="relative h-80 md:h-[440px] map-frame">
                  {loading && <Skeleton className="absolute inset-0" />}
                  {error && (
                    <div className="absolute inset-0 grid place-items-center text-sm text-muted-foreground">
                      Point data unavailable.
                    </div>
                  )}
                  {points && !error && <ClusterZoomMap points={points} color={color} />}
                </div>
                <p className="mt-3 mono-caps leading-relaxed normal-case">
                  Coloured dots are the accidents in this hotspot; grey dots are nearby
                  accidents excluded as noise by the clustering.
                </p>
              </div>

              <div className="flex flex-col">
                <div className="paper-card tone-b self-start px-5 py-3 -rotate-2">
                  <div className="text-5xl font-semibold tracking-[-0.05em] leading-none" style={{ color }}>
                    {cluster.accident_count}
                  </div>
                  <div className="mono-caps mt-1">accidents, 2016–2024</div>
                </div>

                <h4 className="mono-caps mt-7 mb-3 border-b border-[var(--ink)] pb-1">Accident types</h4>
                <div className="flex-1">
                  <AccidentTypeChart data={cluster.accident_types} color={color} />
                </div>

                <Button asChild className="w-full mt-6 rounded-none mono-caps text-[var(--ink)] bg-transparent border-[var(--rule)] hover:bg-[var(--orange)] hover:border-[var(--orange)]" variant="outline">
                  <a href={cluster.google_maps_link} target="_blank" rel="noopener noreferrer">
                    <ExternalLink className="h-4 w-4 mr-2" />
                    Open intersection in Google Maps
                  </a>
                </Button>
              </div>
            </div>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}
