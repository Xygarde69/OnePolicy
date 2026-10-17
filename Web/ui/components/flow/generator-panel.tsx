"use client";

import { Dice5 } from "lucide-react";
import { useEffect, useState } from "react";
import { RobotCanvas } from "@/components/three/robot-canvas";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { api, type GenParams, type Geometry } from "@/lib/api";
import { cn } from "@/lib/utils";

// quadruped_generator.PARAM_KEYS: "scale" is size vs Go2; the rest multiply
// Go2's value at that size. Ranges are the generator's own random ranges.
export const SLIDERS = [
  { key: "scale", label: "Overall size", min: 0.25, max: 4, step: 0.05, unit: "× Go2" },
  { key: "base_length", label: "Body length", min: 0.5, max: 1.5, step: 0.05, unit: "×" },
  { key: "base_width", label: "Body width", min: 0.5, max: 1.5, step: 0.05, unit: "×" },
  { key: "base_height", label: "Body height", min: 0.5, max: 1.5, step: 0.05, unit: "×" },
  { key: "thigh_length", label: "Thigh length", min: 0.4, max: 1.9, step: 0.05, unit: "×" },
  { key: "calf_length", label: "Calf length", min: 0.5, max: 1.5, step: 0.05, unit: "×" },
  { key: "leg_radius", label: "Leg thickness", min: 0.5, max: 1.5, step: 0.05, unit: "×" },
  { key: "hip_link", label: "Hip offset", min: 0.5, max: 1.5, step: 0.05, unit: "×" },
  { key: "density", label: "Body density", min: 0.5, max: 1.5, step: 0.05, unit: "×" },
  { key: "link_mass", label: "Leg masses", min: 0.5, max: 1.5, step: 0.05, unit: "×" },
];

export type Generated = { seed: number; params: GenParams };

/** Seed + sliders; a slider left on "random" keeps the seed's own draw. */
export function GeneratorPanel({ value, onChange }: { value: Generated; onChange: (g: Generated) => void }) {
  const [geometry, setGeometry] = useState<Geometry | null>(null);
  const [error, setError] = useState<string | null>(null);
  const key = JSON.stringify(value);

  useEffect(() => {
    const t = setTimeout(() => {
      api<Geometry>("geometry", { source: "random", seed: value.seed, params: value.params }).then(
        (g) => {
          setGeometry(g);
          setError(null);
        },
        (e) => setError(e.message),
      );
    }, 350);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  const setParam = (k: string, v: number | boolean | undefined) => {
    const params = { ...value.params };
    if (v === undefined) delete params[k];
    else params[k] = v;
    onChange({ ...value, params });
  };

  const xk = value.params.x_knee;
  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_1.1fr]">
      <div className="brutal-lg relative aspect-square overflow-hidden bg-lime lg:sticky lg:top-24 lg:self-start">
        {geometry && <RobotCanvas geometry={geometry} className="absolute inset-0" spin={0.4} />}
        {!geometry && !error && <div className="absolute inset-0 flex items-center justify-center font-heading uppercase">generating…</div>}
        {error && <div className="absolute inset-x-4 bottom-4 rounded-md border-3 border-ink bg-red p-2 text-sm font-bold">{error}</div>}
        {geometry && (
          <div className="absolute top-3 left-3 flex flex-wrap gap-2 font-mono text-[11px] font-bold">
            <span className="rounded-sm border-2 border-ink bg-white px-1.5">{geometry.info.mass.toFixed(1)} kg</span>
            <span className="rounded-sm border-2 border-ink bg-white px-1.5">leg {(geometry.info.leg_length * 100).toFixed(0)} cm</span>
            <span className="rounded-sm border-2 border-ink bg-white px-1.5">{geometry.info.x_knee ? "><" : ">>"} knees</span>
          </div>
        )}
      </div>
      <div className="flex flex-col gap-4">
        <div className="brutal flex flex-wrap items-center gap-3 bg-white p-4">
          <span className="text-sm font-bold uppercase">Seed</span>
          <Input
            type="number"
            value={value.seed}
            onChange={(e) => onChange({ ...value, seed: Math.max(0, Math.floor(Number(e.target.value))) })}
            className="w-36 font-mono"
          />
          <Button variant="pink" size="sm" onClick={() => onChange({ ...value, seed: Math.floor(Math.random() * 2 ** 31) })}>
            <Dice5 /> Roll
          </Button>
          <Button variant="outline" size="sm" onClick={() => onChange({ ...value, params: {} })}>All random</Button>
        </div>
        <div className="brutal flex flex-wrap items-center gap-3 bg-white p-4">
          <span className="text-sm font-bold uppercase">Knees</span>
          {([[undefined, "seed's"], [false, ">> Go2"], [true, ">< ANYmal"]] as const).map(([v, label]) => (
            <button
              key={label}
              type="button"
              onClick={() => setParam("x_knee", v)}
              className={cn("rounded-sm border-2 border-ink px-2 py-1 font-mono text-xs font-bold", xk === v ? "bg-yellow shadow-brutal-sm" : "bg-white")}
            >
              {label}
            </button>
          ))}
        </div>
        {SLIDERS.map((s) => {
          const v = value.params[s.key] as number | undefined;
          const fixed = v !== undefined;
          return (
            <div key={s.key} className={cn("brutal flex flex-col gap-2 p-3", fixed ? "bg-white" : "bg-muted")}>
              <div className="flex items-center gap-3">
                <span className="text-sm font-bold uppercase">{s.label}</span>
                <span className="ml-auto font-mono text-xs font-bold">{fixed ? `${v.toFixed(2)} ${s.unit}` : "random"}</span>
                <Switch checked={fixed} onCheckedChange={(on) => setParam(s.key, on ? 1 : undefined)} />
              </div>
              <Slider
                disabled={!fixed}
                min={s.min}
                max={s.max}
                step={s.step}
                value={[v ?? 1]}
                onValueChange={([nv]) => setParam(s.key, nv)}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}
