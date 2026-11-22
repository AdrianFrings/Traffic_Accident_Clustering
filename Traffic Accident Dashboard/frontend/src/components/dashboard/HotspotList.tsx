import { ArrowUpRight } from 'lucide-react';
import { ScrollArea } from '../ui/scroll-area';
import type { Cluster } from '../../lib/api';

interface Props {
  clusters: Cluster[];
  onSelect: (cluster: Cluster) => void;
}

export function HotspotList({ clusters, onSelect }: Props) {
  if (!clusters.length) {
    return <p className="mono-caps p-4">No hotspots for this city and road user.</p>;
  }

  return (
    <ScrollArea className="h-full">
      <div className="space-y-3 p-1 pr-3">
        {clusters.map((c, i) => (
          <button
            key={c.cluster_id}
            onClick={() => onSelect(c)}
            className="paper-card w-full text-left p-3.5 flex items-start gap-3 group transition-transform hover:-translate-y-0.5 focus-visible:outline-2 focus-visible:outline-[var(--orange)]"
          >
            <span className="rank-num">{String(i + 1).padStart(2, '0')}</span>
            <span className="min-w-0 flex-1">
              <span className="block truncate font-semibold tracking-tight text-[1.02rem]">
                {c.street_name}
              </span>
              <span className="mono-caps block mt-0.5">
                {c.accident_count} accidents
                {c.location_name && c.location_name !== c.street_name ? ` · ${c.location_name}` : ''}
              </span>
            </span>
            <ArrowUpRight className="h-4 w-4 mt-1 text-[var(--muted-ink)] group-hover:text-[var(--orange)] transition-colors" />
          </button>
        ))}
      </div>
    </ScrollArea>
  );
}
