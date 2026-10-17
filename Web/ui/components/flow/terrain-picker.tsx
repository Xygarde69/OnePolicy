"use client";

import { Mountain, Snowflake, Square, Waves, TrendingUp, Grip } from "lucide-react";
import { motion } from "motion/react";
import { toast } from "sonner";
import { ConstructionTag } from "@/components/brutal/under-construction";
import { cn } from "@/lib/utils";

const TERRAINS = [
  { id: "flat", label: "Flat", icon: Square, color: "bg-lime" },
  { id: "stairs", label: "Stairs", icon: TrendingUp, color: "bg-orange" },
  { id: "sand", label: "Sand", icon: Waves, color: "bg-yellow" },
  { id: "rough", label: "Rough", icon: Mountain, color: "bg-pink" },
  { id: "ice", label: "Ice", icon: Snowflake, color: "bg-cyan" },
  { id: "gravel", label: "Gravel", icon: Grip, color: "bg-violet" },
];

export function TerrainPicker({ value, available, onChange }: {
  value: string;
  available: string[];
  onChange: (t: string) => void;
}) {
  return (
    <div className="grid grid-cols-2 gap-4 md:grid-cols-3">
      {TERRAINS.map((t, i) => {
        const ok = available.includes(t.id);
        const Icon = t.icon;
        return (
          <motion.button
            key={t.id}
            type="button"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: i * 0.05 }}
            onClick={() => (ok ? onChange(t.id) : toast(`${t.label} is under construction: only flat ground works for now.`))}
            className={cn(
              "brutal flex flex-col items-start gap-3 p-4 text-left",
              ok ? "brutal-press" : "cursor-not-allowed opacity-70 grayscale-[0.4]",
              value === t.id ? `${t.color} ring-4 ring-ink` : "bg-white",
            )}
          >
            <span className={cn("brutal flex size-12 items-center justify-center", t.color)}>
              <Icon className="size-6" strokeWidth={2.5} />
            </span>
            <span className="font-heading text-xl uppercase">{t.label}</span>
            {!ok && <ConstructionTag />}
          </motion.button>
        );
      })}
    </div>
  );
}
