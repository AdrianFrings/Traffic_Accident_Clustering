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
    <div className="paper-card px-4 py-2 min-w-[7.5rem]">
      <div className="mono-caps">{label}</div>
      <div className="pixel text-2xl leading-tight">{value.toLocaleString('en')}</div>
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
        <label className="mono-caps">City</label>
        <CityCombobox cities={cities} value={city} onChange={onCityChange} />
      </div>

      <div className="flex flex-col gap-1">
        <label className="mono-caps">Road user</label>
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
        <Kpi label="Hotspots" value={totalClusters} />
        <Kpi label="Accidents" value={totalAccidents} />
      </div>
    </div>
  );
}
