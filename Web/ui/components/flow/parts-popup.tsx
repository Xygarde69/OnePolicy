"use client";

import { motion } from "motion/react";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { RobotCanvas } from "@/components/three/robot-canvas";
import { PART_COLORS } from "@/components/three/robot-model";
import { api, type Geometry, type RobotSource } from "@/lib/api";

const PARTS = [
  { part: "base", label: "Base link", color: PART_COLORS.base, stat: (i: Record<string, number>) => `${i.base_mass?.toFixed(2)} kg` },
  { part: "FR_thigh", label: "Thigh", color: PART_COLORS.thigh, stat: (i: Record<string, number>) => `${(i.thigh_length * 100).toFixed(1)} cm` },
  { part: "FR_calf", label: "Calf", color: PART_COLORS.calf, stat: (i: Record<string, number>) => `${(i.calf_length * 100).toFixed(1)} cm` },
];

/** Rings expanding out of the middle of a tile, one after another. */
function Ripples({ color }: { color: string }) {
  return (
    <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
      {[0, 0.8, 1.6].map((d) => (
        <span
          key={d}
          className="animate-ripple absolute size-24 rounded-full border-3 border-ink"
          style={{ animationDelay: `${d}s`, background: color }}
        />
      ))}
    </div>
  );
}

/** Fetches the robot's geometry and pops its base, one thigh and one calf. */
export function useGeometry(source: RobotSource | null) {
  const [geometry, setGeometry] = useState<Geometry | null>(null);
  const [error, setError] = useState<string | null>(null);
  const key = JSON.stringify(source);
  useEffect(() => {
    if (!source) return;
    let live = true;
    setGeometry(null);
    setError(null);
    api<Geometry>("geometry", source).then(
      (g) => live && setGeometry(g),
      (e) => live && setError((e as Error).message),
    );
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return { geometry, error };
}

export function PartsPopup({ open, geometry, onContinue, onOpenChange }: {
  open: boolean;
  geometry: Geometry | null;
  onContinue: () => void;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[92vh] overflow-auto bg-paper sm:max-w-4xl">
        <DialogTitle className="font-heading text-3xl uppercase">
          {geometry ? `Meet ${geometry.name}` : "Building your robot…"}
        </DialogTitle>
        <DialogDescription className="text-base text-ink">
          Adapted to the training convention. Here are the three links every leg and body comes down to.
        </DialogDescription>
        <div className="grid gap-5 sm:grid-cols-3">
          {PARTS.map((p, i) => (
            <motion.div
              key={p.part}
              initial={{ scale: 0, rotate: -12, opacity: 0 }}
              animate={open ? { scale: 1, rotate: 0, opacity: 1 } : {}}
              transition={{ delay: 0.15 + i * 0.18, type: "spring", stiffness: 300, damping: 14 }}
              className="brutal relative flex aspect-[3/4] flex-col overflow-hidden bg-white"
            >
              <div className="relative flex-1 overflow-hidden">
                <Ripples color={`${p.color}66`} />
                {geometry && (
                  <RobotCanvas geometry={geometry} parts={[p.part]} spin={1.1} shadow={false} className="absolute inset-0" />
                )}
              </div>
              <div className="flex items-center justify-between border-t-3 border-ink px-3 py-2" style={{ background: p.color }}>
                <span className="font-heading text-lg uppercase">{p.label}</span>
                {geometry && <span className="font-mono text-xs font-bold">{p.stat(geometry.info)}</span>}
              </div>
            </motion.div>
          ))}
        </div>
        {geometry && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
            className="flex flex-wrap gap-2 font-mono text-xs font-bold"
          >
            {[
              ["mass", `${geometry.info.mass?.toFixed(2)} kg`],
              ["leg", `${(geometry.info.leg_length * 100).toFixed(1)} cm`],
              ["s_L", geometry.info.s_L?.toFixed(2)],
              ["s_m", geometry.info.s_m?.toFixed(2)],
              ["kp knee", `${geometry.info.kp_calf?.toFixed(1)} Nm/rad`],
              ["knees", geometry.info.x_knee ? "><" : ">>"],
            ].map(([k, v]) => (
              <span key={k} className="rounded-sm border-2 border-ink bg-white px-2 py-1">
                {k}: {v}
              </span>
            ))}
          </motion.div>
        )}
        <Button size="lg" className="w-fit" disabled={!geometry} onClick={onContinue}>
          Looks right, continue →
        </Button>
      </DialogContent>
    </Dialog>
  );
}
