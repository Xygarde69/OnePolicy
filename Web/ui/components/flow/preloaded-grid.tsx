"use client";

import { motion } from "motion/react";
import { cn } from "@/lib/utils";

export const PRETTY: Record<string, string> = {
  go2: "Unitree Go2", go1: "Unitree Go1", a1: "Unitree A1", spot: "BD Spot", spot_arm: "Spot + arm",
  anymal_b_xknee: "ANYmal B", anymal_c_xknee: "ANYmal C", barkour_v0: "Barkour v0", barkour_vb: "Barkour vB",
  byte01_v1: "Byte01 v1", byte01_v2: "Byte01 v2", byte01_v2_primitive: "Byte01 v2 (primitive)",
};

export const DESCRIPTIONS: Record<string, string> = {
  go2: "Unitree's small consumer quadruped. Every reward and threshold in training is written in Go2 units.",
  go1: "The Go2's predecessor: nearly the same frame, older actuators.",
  a1: "Unitree's earlier research quadruped, a little lighter than the Go2.",
  spot: "Boston Dynamics' industrial quadruped, much longer legs and heavier than a Go2.",
  spot_arm: "Spot with its arm mounted: extra mass up front that the policy has to carry.",
  anymal_b_xknee: "ANYbotics' ANYmal B. Rear knees point forward (><), unlike the Unitree robots.",
  anymal_c_xknee: "ANYbotics' ANYmal C, the bigger industrial successor of B, also with >< knees.",
  barkour_v0: "Google's agility robot built for dog-style obstacle courses.",
  barkour_vb: "The later Barkour revision, with a reworked body and legs.",
  byte01_v1: "A compact hobby-scale quadruped with mesh-modelled links.",
  byte01_v2: "Byte01's second revision with reworked links.",
  byte01_v2_primitive: "Byte01 v2 with boxes and capsules in place of its meshes: same numbers, simpler shapes.",
};

export const prettyName = (r: string) => PRETTY[r] ?? r;
export const describe = (r: string) => DESCRIPTIONS[r] ?? "Installed from an upload into quadrupeds/.";

const TINTS = ["bg-yellow", "bg-pink", "bg-cyan", "bg-lime", "bg-orange", "bg-violet"];

/** The bundled robots by name only: a robot's meshes load when its name is picked. */
export function PreloadedGrid({ robots, selected, onSelect }: {
  robots: string[];
  selected: string | null;
  onSelect: (robot: string) => void;
}) {
  return (
    <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-4">
      {robots.map((r, i) => (
        <motion.button
          key={r}
          type="button"
          onClick={() => onSelect(r)}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.04 }}
          className={cn(
            "brutal brutal-press group flex items-center gap-3 px-4 py-4 text-left",
            selected === r ? "bg-ink text-paper ring-4 ring-yellow" : TINTS[i % TINTS.length],
          )}
        >
          <span className="flex min-w-0 flex-col">
            <span className="truncate font-heading text-lg uppercase">{prettyName(r)}</span>
            <span className="truncate font-mono text-[10px] font-bold opacity-70">{r}</span>
          </span>
          <span className="ml-auto font-heading text-xl transition-transform group-hover:translate-x-1">→</span>
        </motion.button>
      ))}
    </div>
  );
}
