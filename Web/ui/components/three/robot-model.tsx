"use client";

// One robot (Web/geometry.py's JSON) as a three.js group: every geom in world
// coordinates, flat toon colours by part, ink outlines. MuJoCo is z-up, three
// is y-up, so the whole robot is turned -90 deg about x.

import { Outlines } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useLayoutEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import type { Geom, Geometry } from "@/lib/api";

export const PART_COLORS: Record<string, string> = {
  base: "#ffd23f",
  hip: "#b6f36a",
  thigh: "#ff6fb5",
  calf: "#4dccff",
};

export const partRole = (part: string) => (part === "base" ? "base" : part.split("_")[1] ?? "base");

function decode<T extends Float32Array | Int32Array>(b64: string, Kind: new (buf: ArrayBuffer) => T): T {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Kind(bytes.buffer);
}

const meshCache = new WeakMap<object, Map<number, THREE.BufferGeometry>>();

function meshGeometry(geometry: Geometry, id: number): THREE.BufferGeometry {
  let byId = meshCache.get(geometry.meshes);
  if (!byId) meshCache.set(geometry.meshes, (byId = new Map()));
  const cached = byId.get(id);
  if (cached) return cached;
  const data = geometry.meshes[String(id)];
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(decode(data.vertices, Float32Array), 3));
  g.setIndex(new THREE.BufferAttribute(new Uint32Array(decode(data.faces, Int32Array)), 1));
  g.computeVertexNormals();
  byId.set(id, g);
  return g;
}

// Three's capsule / cylinder run along y; MuJoCo's along z.
const ALONG_Z = new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI / 2, 0, 0));

function primitive(geom: Geom): { geometry: THREE.BufferGeometry; scale?: number[]; turn: boolean } {
  const [a, b, c] = geom.size;
  switch (geom.type) {
    case "box":
      return { geometry: new THREE.BoxGeometry(2 * a, 2 * b, 2 * c), turn: false };
    case "sphere":
      return { geometry: new THREE.SphereGeometry(a, 24, 16), turn: false };
    case "ellipsoid":
      return { geometry: new THREE.SphereGeometry(1, 24, 16), scale: [a, b, c], turn: false };
    case "capsule":
      return { geometry: new THREE.CapsuleGeometry(a, 2 * b, 6, 16), turn: true };
    case "cylinder":
      return { geometry: new THREE.CylinderGeometry(a, a, 2 * b, 24), turn: true };
    default:
      return { geometry: new THREE.BoxGeometry(0.01, 0.01, 0.01), turn: false };
  }
}

type Props = {
  geometry: Geometry;
  /** Only these parts (e.g. ["base"], ["FR_thigh"]); all if omitted. */
  parts?: string[];
  /** Turn about the vertical axis, rad/s. */
  spin?: number;
  outline?: boolean;
};

export function RobotModel({ geometry, parts, spin = 0, outline = true }: Props) {
  const group = useRef<THREE.Group>(null);
  useFrame((_, dt) => {
    if (group.current && spin) group.current.rotation.y += spin * dt;
  });

  const items = useMemo(() => {
    const geoms = parts ? geometry.geoms.filter((g) => parts.includes(g.part)) : geometry.geoms;
    // Size of the shown geoms: every robot is drawn ~1 unit across.
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (const g of geoms) for (let i = 0; i < 3; i++) {
      lo[i] = Math.min(lo[i], g.pos[i] - g.size[0]);
      hi[i] = Math.max(hi[i], g.pos[i] + g.size[0]);
    }
    const extent = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1e-3);
    const center = lo.map((v, i) => (v + hi[i]) / 2);
    return {
      extent,
      center,
      list: geoms.map((g, key) => {
        const quat = new THREE.Quaternion(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
        if (g.type === "mesh" && g.mesh !== undefined) {
          return { key, geometry: meshGeometry(geometry, g.mesh), quat, pos: g.pos, scale: undefined, part: g.part };
        }
        const p = primitive(g);
        if (p.turn) quat.multiply(ALONG_Z);
        return { key, geometry: p.geometry, quat, pos: g.pos, scale: p.scale, part: g.part };
      }),
    };
  }, [geometry, parts]);

  // Normalised to ~1 unit across, centred on the origin. The estimate above
  // is refined from the built meshes' exact bounds once they exist.
  const content = useRef<THREE.Group>(null);
  const [fit, setFit] = useState<{ extent: number; center: number[] } | null>(null);
  useLayoutEffect(() => {
    const g = content.current;
    if (!g) return;
    const box = new THREE.Box3();
    for (const child of g.children) {
      const m = child as THREE.Mesh;
      if (!m.isMesh) continue;
      m.updateMatrix();
      if (!m.geometry.boundingBox) m.geometry.computeBoundingBox();
      box.union(m.geometry.boundingBox!.clone().applyMatrix4(m.matrix));
    }
    if (box.isEmpty()) return;
    const size = box.getSize(new THREE.Vector3());
    setFit({ extent: Math.max(size.x, size.y, size.z, 1e-3), center: box.getCenter(new THREE.Vector3()).toArray() });
  }, [items]);
  const { extent, center } = fit ?? items;
  return (
    <group ref={group}>
      <group rotation={[-Math.PI / 2, 0, 0]} scale={1 / extent}>
        <group ref={content} position={[-center[0], -center[1], -center[2]]}>
          {items.list.map((it) => (
            <mesh
              key={it.key}
              geometry={it.geometry}
              position={it.pos as [number, number, number]}
              quaternion={it.quat}
              scale={it.scale as [number, number, number] | undefined}
              castShadow
            >
              <meshToonMaterial color={PART_COLORS[partRole(it.part)] ?? "#ffffff"} />
              {/* With screenspace off, drei's thickness is in pixels. */}
              {outline && <Outlines thickness={2.5} color="#0b0b0b" />}
            </mesh>
          ))}
        </group>
      </group>
    </group>
  );
}
