import type { CSSProperties, ReactNode } from 'react';

/** Small seeded PRNG so every clipping keeps the same cut between renders. */
function rng(seed: number) {
  let s = seed * 9301 + 49297;
  return () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
}

/** Scissor-cut outline for a clipping, like the clippings on adrianfrings.com. */
export function cutShape(seed: number): string {
  const r = rng(seed);
  const j = (m: number) => Math.round(r() * m * 10) / 10;
  const corners: [number, number][] = [
    [j(5), j(6)],
    [100 - j(5), j(6)],
    [100 - j(5), 100 - j(6)],
    [j(5), 100 - j(6)],
  ];
  const snip = Math.floor(r() * 6); // 0-3 snips that corner, 4-5 none
  const pts: [number, number][] = [];
  corners.forEach(([x, y], i) => {
    if (i !== snip) {
      pts.push([x, y]);
      return;
    }
    const sx = i === 0 || i === 3 ? 8 : -8;
    const sy = i === 0 || i === 1 ? 12 : -12;
    if (i === 0 || i === 2) pts.push([x, y + sy], [x + sx, y]);
    else pts.push([x + sx, y], [x, y + sy]);
  });
  return `polygon(${pts.map(([x, y]) => `${x}% ${y}%`).join(', ')})`;
}

interface ClipProps {
  x: number;
  y: number;
  w: number;
  rot: number;
  tone?: 'a' | 'b' | 'c' | 'ink';
  seed: number;
  z?: number;
  children: ReactNode;
}

function Clip({ x, y, w, rot, tone = 'a', seed, z = 1, children }: ClipProps) {
  const style: CSSProperties = {
    left: `${x}%`,
    top: `${y}%`,
    width: `${w}em`,
    rotate: `${rot}deg`,
    zIndex: z,
    padding: 0,
  };
  return (
    <div className="clip" style={style}>
      <div className={`clip-paper tone-${tone}`} style={{ clipPath: cutShape(seed) }}>
        {children}
      </div>
    </div>
  );
}

function HeatGrid({ seed, cols = 9, rows = 6 }: { seed: number; cols?: number; rows?: number }) {
  const r = rng(seed);
  const cells = Array.from({ length: rows * cols }, (_, k) => {
    const row = Math.floor(k / cols);
    const col = k % cols;
    const hot = row >= 2 && row <= 3 && col >= 3 && col <= 5 ? 0.8 : 0;
    return Math.min(1, 0.1 + r() * 0.6 + hot);
  });
  return (
    <div className="heat" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
      {cells.map((o, i) => (
        <i key={i} style={{ opacity: o }} />
      ))}
    </div>
  );
}

interface Props {
  city: string;
  mode: string;
  hotspots: number;
  accidents: number;
}

/** Newspaper-clipping pile next to the title; the numbers follow the current selection. */
export function HeroCollage({ city, mode, hotspots, accidents }: Props) {
  const who = mode === 'All' ? '' : ` ${mode.toLowerCase()}`;
  return (
    <div className="collage" aria-hidden="true">
      <Clip x={0} y={4} w={13} rot={-5} tone="b" seed={11} z={3}>
        <div className="c-big">{accidents.toLocaleString('en')}</div>
        <div className="c-label">
          {who.trim() ? `${who.trim()} accidents` : 'accidents'} in the {city} hotspots
        </div>
      </Clip>
      <Clip x={48} y={0} w={15} rot={4} tone="c" seed={12} z={2}>
        <div className="c-hed">Fix the street, not the driver.</div>
      </Clip>
      <Clip x={34} y={80} w={17} rot={-3} tone="ink" seed={13} z={4}>
        <div className="c-strip">
          {city} · {hotspots} hotspots
        </div>
      </Clip>
      <Clip x={72} y={40} w={10} rot={9} tone="a" seed={14} z={2}>
        <HeatGrid seed={7} />
        <div className="c-label">Hotspot density</div>
      </Clip>
      <Clip x={8} y={50} w={12} rot={6} tone="c" seed={15} z={1}>
        <div className="c-date">Frankfurt · case study</div>
        <div className="c-hed-s">A bus stop on the median</div>
        <div className="c-lines">
          <i />
          <i />
          <i />
          <i style={{ width: '60%' }} />
        </div>
      </Clip>
    </div>
  );
}
