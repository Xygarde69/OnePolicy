"use client";

import { Gamepad2, MousePointer2, Square } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { Dots } from "@/components/brutal/dots";
import { LimitRow } from "@/components/flow/limits-editor";
import { Button } from "@/components/ui/button";
import { api, type PlayStatus } from "@/lib/api";
import { cn } from "@/lib/utils";

// Keys -> (forward, left, yaw) direction. Arrows turn; W/S forward/back, A/D sideways.
const KEYS: Record<string, [number, number, number]> = {
  w: [1, 0, 0], s: [-1, 0, 0], a: [0, 1, 0], d: [0, -1, 0],
  arrowleft: [0, 0, 1], arrowright: [0, 0, -1], arrowup: [1, 0, 0], arrowdown: [-1, 0, 0],
};
const SEND_HZ = 10;

/** The robot in play.py's viser viewer, driven from the keyboard via /api/drive. */
export function SimView({ start, label }: { start: () => Promise<PlayStatus>; label: string }) {
  const [status, setStatus] = useState<PlayStatus | null>(null);
  const [down, setDown] = useState<Set<string>>(new Set());
  const [drive, setDrive] = useState(true);
  const [speed, setSpeed] = useState(1.0);
  const [lateral, setLateral] = useState(0.5);
  const [yaw, setYaw] = useState(1.0);
  const keys = useRef<Set<string>>(new Set());
  const started = useRef(false);

  useEffect(() => {
    if (started.current) return;
    started.current = true;
    start().then(setStatus, (e) => toast.error(e.message));
  }, [start]);

  useEffect(() => {
    const t = setInterval(() => api<PlayStatus>("play_status").then(setStatus).catch(() => {}), 1500);
    return () => clearInterval(t);
  }, []);

  // Keyboard: held keys only, so letting go stops the robot.
  useEffect(() => {
    const update = (e: KeyboardEvent, on: boolean) => {
      const k = e.key.toLowerCase();
      if (!(k in KEYS) || !drive) return;
      if ((e.target as HTMLElement)?.closest("input, textarea")) return;
      e.preventDefault();
      if (on) keys.current.add(k);
      else keys.current.delete(k);
      setDown(new Set(keys.current));
    };
    const kd = (e: KeyboardEvent) => update(e, true);
    const ku = (e: KeyboardEvent) => update(e, false);
    const blur = () => {
      keys.current.clear();
      setDown(new Set());
    };
    window.addEventListener("keydown", kd);
    window.addEventListener("keyup", ku);
    window.addEventListener("blur", blur);
    return () => {
      window.removeEventListener("keydown", kd);
      window.removeEventListener("keyup", ku);
      window.removeEventListener("blur", blur);
    };
  }, [drive]);

  const command = useCallback((): [number, number, number] => {
    let f = 0, l = 0, y = 0;
    for (const k of keys.current) {
      const [a, b, c] = KEYS[k];
      f += a;
      l += b;
      y += c;
    }
    const clamp = (v: number) => Math.max(-1, Math.min(1, v));
    return [clamp(f) * speed, clamp(l) * lateral, clamp(y) * yaw];
  }, [speed, lateral, yaw]);

  // Commands go out at a fixed rate while the robot runs: play.py stands still
  // if it hears nothing for 0.5 s.
  useEffect(() => {
    if (!status?.running || !status.drive) return;
    const t = setInterval(() => {
      const [forward, left, turn] = command();
      api("drive", { forward, left, yaw: turn }).catch(() => {});
    }, 1000 / SEND_HZ);
    return () => clearInterval(t);
  }, [status?.running, status?.drive, command]);

  const stop = async () => setStatus(await api<PlayStatus>("stop", {}));
  const restart = async () => {
    try {
      setStatus(await start());
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  const port = status?.viser_port;
  const url = port && typeof window !== "undefined" ? `http://${window.location.hostname}:${port}` : null;
  const [f, l, y] = command();
  const crashed = status && !status.running && !status.stopped && status.exit_code !== null && status.exit_code !== 0;

  return (
    <div className="flex flex-col gap-5">
      <div className="grid gap-5 xl:grid-cols-[1fr_20rem]">
        <div className="brutal-lg relative aspect-video overflow-hidden bg-ink">
          {url && status?.running ? (
            <>
              <iframe src={url} className="absolute inset-0 size-full" title="viser viewer" />
              {/* Drive mode: a glass pane keeps the keys on this page, not in the iframe. */}
              {drive && (
                <div
                  tabIndex={0}
                  className="absolute inset-0 cursor-crosshair outline-none"
                  onMouseDown={(e) => (e.currentTarget as HTMLDivElement).focus()}
                />
              )}
              <div className="pointer-events-none absolute top-3 left-3 flex gap-2">
                <span className={cn("rounded-sm border-2 border-ink px-2 py-0.5 font-mono text-xs font-bold uppercase", drive ? "bg-yellow" : "bg-white")}>
                  {drive ? "drive mode" : "camera mode"}
                </span>
                <span className="rounded-sm border-2 border-ink bg-white px-2 py-0.5 font-mono text-xs font-bold">{label}</span>
              </div>
            </>
          ) : (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 p-6 text-center text-paper">
              {crashed ? (
                <>
                  <span className="font-heading text-3xl text-red uppercase">play.py stopped</span>
                  <pre className="max-h-60 w-full overflow-auto rounded-md border-3 border-red bg-ink p-3 text-left font-mono text-[11px] text-paper/80">{status?.log}</pre>
                  <Button onClick={restart}>Try again</Button>
                </>
              ) : status && !status.running && (status.stopped || status.exit_code !== null) ? (
                <>
                  <span className="font-heading text-3xl uppercase">Stopped</span>
                  <Button onClick={restart}>Start again</Button>
                </>
              ) : (
                <>
                  <span className="font-heading text-4xl uppercase">
                    Placing the robot<Dots />
                  </span>
                  <span className="font-mono text-xs text-paper/60">
                    fixing + adapting the MJCF, loading the policy, compiling MuJoCo-Warp kernels
                  </span>
                  <span className="hazard animate-stripes h-4 w-64 rounded-sm border-3 border-paper" />
                </>
              )}
            </div>
          )}
        </div>

        <div className="flex flex-col gap-4">
          <div className="brutal flex flex-col items-center gap-3 bg-white p-4">
            <span className="font-heading text-lg uppercase">Drive</span>
            <div className="grid grid-cols-3 gap-1.5">
              <span />
              <span className="kbd" data-down={down.has("w") || down.has("arrowup")}>W</span>
              <span />
              <span className="kbd" data-down={down.has("a")}>A</span>
              <span className="kbd" data-down={down.has("s") || down.has("arrowdown")}>S</span>
              <span className="kbd" data-down={down.has("d")}>D</span>
            </div>
            <div className="flex gap-1.5">
              <span className="kbd" data-down={down.has("arrowleft")}>←</span>
              <span className="kbd" data-down={down.has("arrowright")}>→</span>
            </div>
            <span className="text-center text-xs font-medium">W/S forward · A/D sideways · ←/→ turn</span>
            <div className="grid w-full grid-cols-3 gap-2 font-mono text-[11px] font-bold">
              {[["fwd", f, "m/s"], ["left", l, "m/s"], ["yaw", y, "rad/s"]].map(([k, v, u]) => (
                <div key={k as string} className={cn("rounded-sm border-2 border-ink p-1 text-center", v !== 0 ? "bg-lime" : "bg-muted")}>
                  {k}
                  <div className="text-sm">{(v as number).toFixed(2)}</div>
                  <div className="text-[9px] text-ink/60">{u}</div>
                </div>
              ))}
            </div>
            <span className="text-center font-mono text-[10px] text-ink/60">Go2 units, scaled to the robot&apos;s size</span>
          </div>
          <div className="flex gap-2">
            <Button variant={drive ? "default" : "outline"} className="flex-1" onClick={() => setDrive(true)}>
              <Gamepad2 /> Drive
            </Button>
            <Button variant={!drive ? "cyan" : "outline"} className="flex-1" onClick={() => setDrive(false)}>
              <MousePointer2 /> Camera
            </Button>
          </div>
          <Button variant="destructive" disabled={!status?.running} onClick={stop}>
            <Square /> Stop the sim
          </Button>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <LimitRow limit={{ label: "Forward speed", min: 0.1, max: 3, step: 0.1, unit: "m/s", value: 1 }} value={speed} onChange={setSpeed} />
        <LimitRow limit={{ label: "Sideways speed", min: 0.1, max: 1.5, step: 0.1, unit: "m/s", value: 0.5 }} value={lateral} onChange={setLateral} />
        <LimitRow limit={{ label: "Turn rate", min: 0.1, max: 2, step: 0.1, unit: "rad/s", value: 1 }} value={yaw} onChange={setYaw} />
      </div>

      <details className="brutal bg-white">
        <summary className="cursor-pointer p-3 text-sm font-bold uppercase">play.py output</summary>
        <pre className="max-h-72 overflow-auto border-t-3 border-ink bg-ink p-3 font-mono text-[11px] text-lime">
          {status?.command}
          {"\n\n"}
          {status?.log}
        </pre>
      </details>
    </div>
  );
}
