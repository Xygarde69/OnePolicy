"use client";

import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import type { Limit } from "@/lib/api";
import { cn } from "@/lib/utils";

const decimals = (step: number) => Math.max(0, -Math.floor(Math.log10(step)));

export function LimitRow({ limit, value, onChange }: {
  limit: Pick<Limit, "label" | "min" | "max" | "step" | "unit" | "value">;
  value: number;
  onChange: (v: number) => void;
}) {
  const changed = value !== limit.value;
  // Learning rate spans three decades: slide it on a log scale.
  const log = limit.max / Math.max(limit.min, 1e-12) >= 100;
  const toSlider = (v: number) => (log ? Math.log10(v) : v);
  const fromSlider = (v: number) => (log ? Number((10 ** v).toPrecision(2)) : v);
  return (
    <div className={cn("brutal flex flex-col gap-3 bg-white p-4", changed && "bg-yellow/40")}>
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-sm font-bold uppercase">{limit.label}</span>
        <span className="font-mono text-[11px] text-ink/60">default {limit.value}{limit.unit && ` ${limit.unit}`}</span>
      </div>
      <div className="flex items-center gap-3">
        <Slider
          min={toSlider(limit.min)}
          max={toSlider(limit.max)}
          step={log ? 0.01 : limit.step}
          value={[toSlider(value)]}
          onValueChange={([v]) => onChange(fromSlider(v))}
        />
        <Input
          type="number"
          step={limit.step}
          min={limit.min}
          max={limit.max}
          value={log ? value : Number(value.toFixed(decimals(limit.step)))}
          onChange={(e) => onChange(Math.min(limit.max, Math.max(limit.min, Number(e.target.value))))}
          className="h-9 w-28 font-mono text-xs"
        />
      </div>
    </div>
  );
}

export function LimitsEditor({ limits, values, onChange }: {
  limits: Limit[];
  values: Record<string, number>;
  onChange: (values: Record<string, number>) => void;
}) {
  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {limits.map((l) => (
        <LimitRow key={l.key} limit={l} value={values[l.key] ?? l.value} onChange={(v) => onChange({ ...values, [l.key]: v })} />
      ))}
    </div>
  );
}
