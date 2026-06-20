import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../ui/select';
import { CityCombobox } from './CityCombobox';

interface Props {
  cities: string[];
  modes: string[];
  city: string;
  mode: string;
  totalClusters: number;
  totalAccidents: number;
  onCityChange: (city: string) => void;
  onModeChange: (mode: string) => void;
}

function Kpi({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-lg border border-border bg-card px-4 py-2">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="text-xl font-bold">{value.toLocaleString()}</div>
    </div>
  );
}

export function Controls({
  cities,
  modes,
  city,
  mode,
  totalClusters,
  totalAccidents,
  onCityChange,
  onModeChange,
}: Props) {
  return (
    <div className="flex flex-wrap items-end gap-3">
      <div className="flex flex-col gap-1">
        <label className="text-xs text-muted-foreground">City</label>
        <CityCombobox cities={cities} value={city} onChange={onCityChange} />
      </div>

      <div className="flex flex-col gap-1">
        <label className="text-xs text-muted-foreground">Participant</label>
        <Select value={mode} onValueChange={onModeChange}>
          <SelectTrigger className="w-[160px]">
            <SelectValue placeholder="Mode" />
          </SelectTrigger>
          <SelectContent>
            {modes.map((m) => (
              <SelectItem key={m} value={m}>
                {m}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="flex gap-3 ml-auto">
        <Kpi label="Clusters" value={totalClusters} />
        <Kpi label="Accidents" value={totalAccidents} />
      </div>
    </div>
  );
}
