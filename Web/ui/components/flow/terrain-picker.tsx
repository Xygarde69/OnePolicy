"use client";

import { ArrowDownRight, ArrowUpRight, Mountain, Square, Waves } from "lucide-react";
import { motion } from "motion/react";
import { OnePolicy } from "@/components/brutal/brand";
import { cn } from "@/lib/utils";

// Ids match server.py's TERRAINS. Every terrain but flat is a training terrain
// of the rough task at terrain level 4, scaled to the robot's size and walked
// by OnePolicy_r.
const TERRAINS = [
  { id: "flat", label: "Flat", icon: Square, color: "bg-lime" },
  { id: "slope_up", label: "Slope up", icon: ArrowUpRight, color: "bg-yellow" },
  { id: "slope_down", label: "Slope down", icon: ArrowDownRight, color: "bg-yellow" },
  { id: "rough", label: "Rough", icon: Mountain, color: "bg-pink" },
  { id: "waves", label: "Waves", icon: Waves, color: "bg-cyan" },
];

export function TerrainPicker({ value, available, onChange }: {
  value: string;
  available: string[];
  onChange: (t: string) => void;
}) {
  return (
    <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
      {TERRAINS.filter((t) => available.includes(t.id)).map((t, i) => {
        const Icon = t.icon;
        return (
          <motion.button
            key={t.id}
            type="button"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: i * 0.05 }}
            onClick={() => onChange(t.id)}
            className={cn(
              "brutal brutal-press flex flex-col items-start gap-3 p-4 text-left",
              value === t.id ? `${t.color} ring-4 ring-ink` : "bg-white",
            )}
          >
            <span className={cn("brutal flex size-12 items-center justify-center", t.color)}>
              <Icon className="size-6" strokeWidth={2.5} />
            </span>
            <span className="font-heading text-xl uppercase">{t.label}</span>
            <span className="font-mono text-[10px] font-bold uppercase">
              <OnePolicy r={t.id !== "flat"} className="text-sm" />
            </span>
          </motion.button>
        );
      })}
    </div>
  );
}
