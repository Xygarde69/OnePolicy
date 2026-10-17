"use client";

import { useEffect, useState } from "react";
import { api, type Geometry } from "@/lib/api";
import { RobotCanvas } from "./robot-canvas";

/** A generated robot spinning on the landing page; a new seed per click. */
export function HeroRobot() {
  const [seed, setSeed] = useState(7);
  const [geometry, setGeometry] = useState<Geometry | null>(null);
  useEffect(() => {
    api<Geometry>("geometry", { source: "random", seed }).then(setGeometry, () => setGeometry(null));
  }, [seed]);

  return (
    <div className="brutal-lg relative aspect-square w-full overflow-hidden bg-cyan">
      <div className="absolute inset-0 bg-[linear-gradient(var(--ink)_1px,transparent_1px),linear-gradient(90deg,var(--ink)_1px,transparent_1px)] bg-[size:40px_40px] opacity-15" />
      {geometry ? (
        <RobotCanvas geometry={geometry} className="absolute inset-0" spin={0.5} />
      ) : (
        <div className="absolute inset-0 flex items-center justify-center font-heading text-2xl uppercase">
          generating…
        </div>
      )}
      <button
        type="button"
        onClick={() => setSeed(Math.floor(Math.random() * 1e6))}
        className="brutal brutal-press absolute bottom-4 left-4 bg-white px-3 py-1 font-mono text-xs font-bold uppercase"
      >
        ↻ new robot #{seed}
      </button>
      {geometry && (
        <span className="brutal absolute top-4 right-4 bg-yellow px-2 py-1 font-mono text-xs font-bold">
          {geometry.info.mass?.toFixed(1)} kg · leg {(geometry.info.leg_length * 100).toFixed(0)} cm
        </span>
      )}
    </div>
  );
}
