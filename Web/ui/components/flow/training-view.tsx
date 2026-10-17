"use client";

import { Square, Terminal } from "lucide-react";
import { motion } from "motion/react";
import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Dots } from "@/components/brutal/dots";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import type { TrainPoint, TrainStatus } from "@/lib/api";
import { cn } from "@/lib/utils";

const BONUS = "#3a62d9";
const PENALTY = "#d9367f";

const PHASE_TEXT: Record<TrainStatus["phase"], string> = {
  idle: "Ready",
  generating: "Generating robots",
  building: "Building the sim",
  training: "Training",
  done: "Done!",
  stopped: "Stopped",
  failed: "Failed",
};

function Letters({ text }: { text: string }) {
  return (
    <span className="inline-flex">
      {text.split("").map((ch, i) => (
        <motion.span
          key={`${text}-${i}`}
          animate={{ y: [0, -14, 0] }}
          transition={{ duration: 0.9, repeat: Infinity, delay: i * 0.08, ease: "easeInOut" }}
          className="inline-block"
        >
          {ch === " " ? "\u00a0" : ch}
        </motion.span>
      ))}
    </span>
  );
}

/** The big animated "TRAINING..." panel; press X for the stats. */
export function TrainingView({ status, onStop }: { status: TrainStatus; onStop: () => void }) {
  const [stats, setStats] = useState(false);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = (e.target as HTMLElement)?.closest("input, textarea, [contenteditable]");
      if (!typing && e.key.toLowerCase() === "x") setStats((s) => !s);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const it = status.latest.iteration ?? 0;
  const total = status.total ?? 0;
  const pct = total ? Math.min(100, (100 * (it + 1)) / total) : 0;
  const live = status.running;
  const bg = status.phase === "failed" ? "bg-red" : status.phase === "done" ? "bg-lime" : "bg-pink";

  return (
    <div className={cn("brutal-lg relative flex flex-col items-center gap-6 overflow-hidden px-6 py-12 text-center", bg)}>
      <div className="absolute inset-0 bg-[repeating-linear-gradient(-45deg,transparent_0_22px,rgba(0,0,0,0.05)_22px_44px)]" />
      <div className="relative font-heading text-5xl uppercase sm:text-7xl">
        {live ? <Letters text={PHASE_TEXT[status.phase]} /> : PHASE_TEXT[status.phase]}
        {live && <Dots />}
      </div>
      {status.phase === "training" || total ? (
        <div className="relative w-full max-w-2xl">
          <div className="h-10 overflow-hidden rounded-md border-3 border-ink bg-white">
            <motion.div
              className="hazard animate-stripes h-full border-r-3 border-ink"
              animate={{ width: `${pct}%` }}
              transition={{ type: "spring", stiffness: 80, damping: 20 }}
            />
          </div>
          <div className="mt-2 flex justify-between font-mono text-sm font-bold">
            <span>iteration {it + 1} / {total}</span>
            <span>reward {status.latest["Mean reward"]?.toFixed(2) ?? "–"}</span>
            <span>ETA {fmtEta(status)}</span>
          </div>
        </div>
      ) : (
        <p className="relative max-w-lg font-medium">
          train.py is generating one robot per env and compiling the batched MuJoCo-Warp sim. This takes a minute.
        </p>
      )}
      <div className="relative flex flex-wrap items-center justify-center gap-3">
        <button
          type="button"
          onClick={() => setStats(true)}
          className="brutal brutal-press flex items-center gap-2 bg-white px-4 py-2 font-bold uppercase"
        >
          press <span className="kbd h-8 min-w-8">X</span> to see stats
        </button>
        {live && (
          <Button variant="ink" onClick={onStop}>
            <Square /> Stop (keeps saved checkpoints)
          </Button>
        )}
      </div>
      <StatsOverlay open={stats} onOpenChange={setStats} status={status} />
    </div>
  );
}

const fmtEta = (status: TrainStatus) => status.eta ?? "–";

function Tile({ label, value, sub, color = "bg-white" }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className={cn("brutal flex flex-col gap-1 p-3", color)}>
      <span className="font-mono text-[11px] font-bold uppercase">{label}</span>
      <span className="font-heading text-2xl leading-none">{value}</span>
      {sub && <span className="font-mono text-[11px] text-ink/60">{sub}</span>}
    </div>
  );
}

function SeriesChart({ data, field, title, unit }: { data: TrainPoint[]; field: keyof TrainPoint; title: string; unit?: string }) {
  return (
    <div className="brutal flex flex-col gap-2 bg-white p-3">
      <span className="font-heading text-sm uppercase">{title}</span>
      <div className="h-44">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: -12 }}>
            <CartesianGrid stroke="#0b0b0b" strokeOpacity={0.08} vertical={false} />
            <XAxis dataKey="iteration" tick={{ fontSize: 11, fontFamily: "var(--font-code)" }} stroke="#0b0b0b" strokeOpacity={0.4} />
            <YAxis tick={{ fontSize: 11, fontFamily: "var(--font-code)" }} stroke="#0b0b0b" strokeOpacity={0.4} domain={["auto", "auto"]} />
            <Tooltip
              contentStyle={{ border: "3px solid #0b0b0b", borderRadius: 6, boxShadow: "3px 3px 0 #0b0b0b", fontFamily: "var(--font-code)", fontSize: 12 }}
              labelFormatter={(v) => `iteration ${v}`}
              formatter={(v) => [`${Number(v).toFixed(3)}${unit ?? ""}`, title]}
              cursor={{ stroke: "#0b0b0b", strokeWidth: 1, strokeDasharray: "4 4" }}
            />
            <Line type="monotone" dataKey={field} stroke="#0b0b0b" strokeWidth={2} dot={false} isAnimationActive={false} activeDot={{ r: 5, stroke: "#fff", strokeWidth: 2, fill: "#0b0b0b" }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function StatsOverlay({ open, onOpenChange, status }: { open: boolean; onOpenChange: (o: boolean) => void; status: TrainStatus }) {
  const [log, setLog] = useState(false);
  const l = status.latest;
  const terms = Object.entries(l)
    .filter(([k]) => k.startsWith("Episode_Reward/"))
    .map(([k, v]) => ({ name: k.replace("Episode_Reward/", ""), value: v }))
    .sort((a, b) => b.value - a.value);
  const elapsed = status.started ? Math.round(Date.now() / 1000 - status.started) : 0;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[94vh] overflow-auto bg-paper text-left sm:max-w-6xl">
        <div className="flex items-center gap-3">
          <span className={cn("size-3 rounded-full border-2 border-ink", status.running ? "animate-pulse bg-lime" : "bg-muted")} />
          <DialogTitle className="font-heading text-2xl uppercase">Live stats</DialogTitle>
          <span className="ml-auto font-mono text-xs">press X to close · refreshes every 2 s</span>
        </div>
        <DialogDescription className="font-mono text-xs break-all text-ink/70">{status.log_dir ?? status.command}</DialogDescription>

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
          <Tile label="Iteration" value={`${(l.iteration ?? -1) + 1}`} sub={`of ${status.total ?? "?"}`} color="bg-yellow" />
          <Tile label="Mean reward" value={l["Mean reward"]?.toFixed(2) ?? "–"} sub="last 100 episodes" />
          <Tile label="Episode length" value={l["Mean episode length"]?.toFixed(0) ?? "–"} sub="steps (50 Hz)" />
          <Tile label="Steps / s" value={l["Steps per second"] ? Math.round(l["Steps per second"]).toLocaleString() : "–"} />
          <Tile label="Action std" value={l["Mean action std"]?.toFixed(2) ?? "–"} sub="exploration noise" />
          <Tile label="Elapsed" value={`${Math.floor(elapsed / 60)}m ${elapsed % 60}s`} sub={`ETA ${fmtEta(status)}`} />
        </div>

        {status.history.length > 1 ? (
          <div className="grid gap-3 lg:grid-cols-3">
            <SeriesChart data={status.history} field="reward" title="Mean reward" />
            <SeriesChart data={status.history} field="episode_length" title="Mean episode length" />
            <SeriesChart data={status.history} field="std" title="Action noise std" />
          </div>
        ) : (
          <div className="brutal bg-white p-6 text-center font-heading uppercase">
            Charts appear after the second iteration<Dots />
          </div>
        )}

        {terms.length > 0 && (
          <div className="brutal flex flex-col gap-2 bg-white p-3">
            <div className="flex flex-wrap items-center gap-4">
              <span className="font-heading text-sm uppercase">Reward terms this iteration (per episode)</span>
              <span className="flex items-center gap-1.5 font-mono text-xs"><span className="size-3 rounded-sm" style={{ background: BONUS }} /> bonus</span>
              <span className="flex items-center gap-1.5 font-mono text-xs"><span className="size-3 rounded-sm" style={{ background: PENALTY }} /> penalty</span>
            </div>
            <div style={{ height: 24 * terms.length + 30 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={terms} layout="vertical" margin={{ top: 0, right: 40, bottom: 0, left: 10 }} barCategoryGap={2}>
                  <XAxis type="number" tick={{ fontSize: 11, fontFamily: "var(--font-code)" }} stroke="#0b0b0b" strokeOpacity={0.4} />
                  <YAxis type="category" dataKey="name" width={170} tick={{ fontSize: 11, fontFamily: "var(--font-code)", fill: "#0b0b0b" }} stroke="#0b0b0b" strokeOpacity={0.4} />
                  <ReferenceLine x={0} stroke="#0b0b0b" strokeWidth={2} />
                  <Tooltip
                    contentStyle={{ border: "3px solid #0b0b0b", borderRadius: 6, boxShadow: "3px 3px 0 #0b0b0b", fontFamily: "var(--font-code)", fontSize: 12 }}
                    formatter={(v) => [Number(v).toFixed(4), "per episode"]}
                    cursor={{ fill: "#0b0b0b", fillOpacity: 0.05 }}
                  />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]} isAnimationActive={false} label={{ position: "right", fontSize: 10, fontFamily: "var(--font-code)", fill: "#0b0b0b", formatter: (v: unknown) => Number(v).toFixed(3) }}>
                    {terms.map((t) => <Cell key={t.name} fill={t.value >= 0 ? BONUS : PENALTY} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        <Button variant="outline" size="sm" className="w-fit" onClick={() => setLog((s) => !s)}>
          <Terminal /> {log ? "Hide" : "Show"} train.py output
        </Button>
        {log && (
          <pre className="brutal max-h-80 overflow-auto bg-ink p-3 font-mono text-[11px] text-lime">{status.log}</pre>
        )}
      </DialogContent>
    </Dialog>
  );
}
