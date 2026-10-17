"use client";

import { ContactShadows, OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import type { Geometry } from "@/lib/api";
import { RobotModel } from "./robot-model";

export function Lights() {
  return (
    <>
      <ambientLight intensity={0.9} />
      <directionalLight position={[2, 3, 2]} intensity={1.4} />
      <directionalLight position={[-2, 1, -1]} intensity={0.3} />
    </>
  );
}

type Props = {
  geometry: Geometry;
  parts?: string[];
  spin?: number;
  controls?: boolean;
  shadow?: boolean;
  className?: string;
};

/** A self-contained canvas with one robot, ~1 unit across, seen from the front-left. */
export function RobotCanvas({ geometry, parts, spin = 0.6, controls = true, shadow = true, className }: Props) {
  return (
    <div className={className}>
      {/* offsetSize: measure without CSS transforms (popups scale in from 0). */}
      <Canvas camera={{ position: [1.4, 0.8, 1.65], fov: 35 }} dpr={[1, 2]} resize={{ offsetSize: true }}>
        <Lights />
        <RobotModel geometry={geometry} parts={parts} spin={spin} />
        {shadow && <ContactShadows position={[0, -0.45, 0]} opacity={0.5} scale={3} blur={2.2} far={1} />}
        {controls && <OrbitControls enablePan={false} minDistance={0.8} maxDistance={4} />}
      </Canvas>
    </div>
  );
}
