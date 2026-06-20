import { ChevronRight } from 'lucide-react';
import { ScrollArea } from '../ui/scroll-area';
import type { Cluster } from '../../lib/api';
import { modeColor } from '../../lib/map';

interface Props {
  clusters: Cluster[];
  mode: string;
  onSelect: (cluster: Cluster) => void;
}

export function HotspotList({ clusters, mode, onSelect }: Props) {
  const color = modeColor(mode);

  if (!clusters.length) {
    return (
      <p className="text-sm text-muted-foreground p-4">
        No clusters found for this city and mode.
      </p>
    );
  }

  return (
    <ScrollArea className="h-full">
      <div className="space-y-2 p-1">
        {clusters.map((c, i) => (
          <button
            key={c.cluster_id}
            onClick={() => onSelect(c)}
            className="w-full text-left rounded-lg border border-border bg-card hover:bg-accent transition-colors p-3 flex items-center gap-3 group"
          >
            <span
              className="flex-shrink-0 h-7 w-7 grid place-items-center rounded-full text-xs font-semibold text-white"
              style={{ backgroundColor: color }}
            >
              {i + 1}
            </span>
            <span className="min-w-0 flex-1">
              <span className="block truncate font-medium text-sm">{c.street_name}</span>
              <span className="block text-xs text-muted-foreground">
                {c.accident_count} accidents
                {c.location_name && c.location_name !== c.street_name
                  ? ` · ${c.location_name}`
                  : ''}
              </span>
            </span>
            <ChevronRight className="h-4 w-4 text-muted-foreground group-hover:translate-x-0.5 transition-transform" />
          </button>
        ))}
      </div>
    </ScrollArea>
  );
}
