import {
  Bar,
  BarChart,
  Cell,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { AccidentType } from '../../lib/api';

interface Props {
  data: AccidentType[];
  color: string;
}


export function AccidentTypeChart({ data, color }: Props) {
  if (!data.length) {
    return <p className="text-sm text-muted-foreground">No type breakdown available.</p>;
  }

  const sorted = [...data].sort((a, b) => b.count - a.count);
  const max = Math.max(...sorted.map((d) => d.count));

  return (
    <ResponsiveContainer width="100%" height={Math.max(200, sorted.length * 40)}>
      <BarChart
        data={sorted}
        layout="vertical"
        barCategoryGap="22%"
        margin={{ left: 4, right: 36, top: 4, bottom: 4 }}
      >
        {/* Headroom on the axis so the value label at the bar end isn't clipped. */}
        <XAxis type="number" hide domain={[0, max * 1.15]} />
        <YAxis
          type="category"
          dataKey="label"
          width={210}
          interval={0}
          tick={{ fill: 'var(--foreground)', fontSize: 12, fontFamily: 'Geist' }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          cursor={{ fill: 'var(--accent)', opacity: 0.4 }}
          contentStyle={{
            background: 'var(--popover)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius)',
            color: 'var(--popover-foreground)',
            fontSize: 12,
          }}
          formatter={(value: number) => [value, 'Accidents']}
        />
        <Bar dataKey="count" radius={0} minPointSize={3}>
          {sorted.map((_, i) => (
            <Cell key={i} fill={color} fillOpacity={1 - i * (0.45 / sorted.length)} />
          ))}
          <LabelList
            dataKey="count"
            position="right"
            style={{ fill: 'var(--foreground)', fontSize: 12, fontFamily: 'Geist Mono' }}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
