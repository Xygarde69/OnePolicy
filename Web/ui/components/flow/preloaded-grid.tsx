"use client";

import { PerspectiveCamera, View } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { motion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { Lights } from "@/components/three/robot-canvas";
import { RobotModel } from "@/components/three/robot-model";
import { api, type Geometry } from "@/lib/api";
import { cn } from "@/lib/utils";

const PRETTY: Record<string, string> = {
  go2: "Unitree Go2", go1: "Unitree Go1", a1: "Unitree A1", spot: "BD Spot", spot_arm: "Spot + arm",
  anymal_b_xknee: "ANYmal B", anymal_c_xknee: "ANYmal C", barkour_v0: "Barkour v0", barkour_vb: "Barkour vB",
  byte01_v1: "Byte01 v1", byte01_v2: "Byte01 v2", byte01_v2_primitive: "Byte01 v2 (primitive)",
};
const TINTS = ["bg-yellow", "bg-pink", "bg-cyan", "bg-lime", "bg-orange", "bg-violet"];

/** Every bundled robot spinning in its card; one shared WebGL canvas (drei View). */
export function PreloadedGrid({ robots, selected, onSelect }: {
  robots: string[];
  selected: string | null;
  onSelect: (robot: string) => void;
}) {
  const container = useRef<HTMLDivElement>(null!);
  const [geoms, setGeoms] = useState<Record<string, Geometry | "error">>({});

  // One at a time: some meshes are several MB.
  useEffect(() => {
    let live = true;
    (async () => {
      for (const r of robots) {
        try {
          const g = await api<Geometry>("geometry", { source: "robot", robot: r });
          if (live) setGeoms((m) => ({ ...m, [r]: g }));
        } catch {
          if (live) setGeoms((m) => ({ ...m, [r]: "error" }));
        }
      }
    })();
    return () => {
      live = false;
    };
  }, [robots]);

  return (
    <div ref={container} className="relative">
      <div className="grid grid-cols-2 gap-5 md:grid-cols-3 lg:grid-cols-4">
        {robots.map((r, i) => {
          const g = geoms[r];
          return (
            <motion.button
              key={r}
              type="button"
              onClick={() => onSelect(r)}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.04 }}
              className={cn(
                "brutal brutal-press flex flex-col overflow-hidden text-left",
                selected === r ? "bg-ink text-paper ring-4 ring-yellow" : "bg-white",
              )}
            >
              <div className={cn("relative aspect-square border-b-3 border-ink", TINTS[i % TINTS.length])}>
                {g && g !== "error" ? (
                  <View className="absolute inset-0">
                    <PerspectiveCamera makeDefault position={[1.4, 0.8, 1.65]} fov={35} onUpdate={(c) => c.lookAt(0, 0, 0)} />
                    <Lights />
                    <RobotModel geometry={g} spin={0.7} />
                  </View>
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center font-mono text-xs font-bold uppercase">
                    {g === "error" ? "can't load" : <span className="size-6 animate-spin rounded-sm border-3 border-ink border-t-white" />}
                  </div>
                )}
              </div>
              <div className="flex items-center justify-between gap-2 px-3 py-2">
                <span className="font-heading text-sm uppercase">{PRETTY[r] ?? r}</span>
                {g && g !== "error" && (
                  <span className="font-mono text-[10px] font-bold">{g.info.mass.toFixed(1)} kg</span>
                )}
              </div>
            </motion.button>
          );
        })}
      </div>
      <Canvas
        eventSource={container}
        className="!pointer-events-none !fixed !inset-0"
        style={{ position: "fixed", inset: 0, pointerEvents: "none" }}
        dpr={[1, 2]}
      >
        <View.Port />
      </Canvas>
    </div>
  );
}
