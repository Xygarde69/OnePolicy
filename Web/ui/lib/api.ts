// Client for the Python harness (Web/server.py). Every endpoint takes and
// returns JSON; errors come back as {error} with a 4xx/5xx status.

export class ApiError extends Error {}

export async function api<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`/api/${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  let data: unknown;
  try {
    data = await res.json();
  } catch {
    throw new ApiError(`Server answered ${res.status} without JSON. Is server.py running?`);
  }
  if (!res.ok) throw new ApiError((data as { error?: string }).error ?? `HTTP ${res.status}`);
  return data as T;
}

export const downloadUrl = (path: string) => `/api/download?path=${encodeURIComponent(path)}`;

// ------------------------------------------------------------------ types

export type State = {
  robots: string[];
  checkpoints: string[];
  onepolicy: string | null;
  onepolicy_r: string | null;
  terrains: string[];
};

export type Geom = {
  type: "sphere" | "capsule" | "ellipsoid" | "cylinder" | "box" | "mesh";
  part: string; // "base" or "<leg>_<hip|thigh|calf>"
  size: number[];
  pos: number[];
  quat: number[]; // MuJoCo order: w, x, y, z
  rgba: number[];
  mesh?: number;
};

export type Geometry = {
  name: string;
  seed?: number;
  geoms: Geom[];
  meshes: Record<string, { vertices: string; faces: string }>;
  bounds: [number[], number[]];
  info: Record<string, number>;
  fixes?: string[];
};

export type RobotSource =
  | { source: "robot"; robot: string }
  | { source: "random"; seed: number; params?: GenParams }
  | { source: "upload"; upload: string; xml: string; forward?: string; xknee?: boolean };

export type GenParams = Partial<Record<string, number | boolean>>;

export type JointRow = {
  index: number;
  name: string;
  type: string;
  body: string;
  axis: string;
  range: string;
  guess: string;
  ok: boolean;
};

export type Inspect = {
  forward: string;
  xknee: boolean;
  file: string;
  table: { file: string; joints: number; bodies: number; mass: number; forward: string; legs: number; rows: JointRow[] } | null;
  table_error?: string;
  errors: string[];
  report: string[];
  pose: Record<string, number> | null;
  text?: string;
  default_name?: string;
};

export type Upload = { upload: string; xmls: string[] };

export type RewardTerm = {
  name: string;
  weight: number;
  func: string;
  doc: string;
  params: Record<string, number>;
  curriculum: { step: number; weight: number }[] | null;
};

export type Limit = {
  key: string;
  label: string;
  flag: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
};

export type RewardTable = {
  terms: RewardTerm[];
  limits: Limit[];
  variables: Record<string, string>;
  functions: string[];
  gen_params: string[];
};

export type ConditionTest = {
  kept: number;
  samples: number;
  rows: { seed: number; ok: boolean; mass: number; leg_length: number }[];
};

export type TrainPoint = {
  iteration: number;
  reward: number | null;
  episode_length: number | null;
  std: number | null;
  fps: number | null;
};

export type TrainStatus = {
  running: boolean;
  phase: "idle" | "generating" | "building" | "training" | "done" | "stopped" | "failed";
  exit_code: number | null;
  command: string | null;
  started: number | null;
  log_dir: string | null;
  checkpoints: string[];
  log: string;
  history: TrainPoint[];
  latest: Record<string, number>;
  total: number | null;
  eta: string | null;
};

export type PlayStatus = {
  running: boolean;
  exit_code: number | null;
  viser_port: number | null;
  drive: boolean;
  /** Ended by Stop (not a crash). */
  stopped: boolean;
  command: string | null;
  log: string;
};

// ------------------------------------------------------------- uploads

export type PickedFile = { path: string; file: File };

async function toBase64(file: File): Promise<string> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

export async function uploadFiles(
  files: PickedFile[],
  onProgress?: (done: number, total: number) => void,
): Promise<Upload> {
  const payload = [];
  for (let i = 0; i < files.length; i++) {
    payload.push({ path: files[i].path, data: await toBase64(files[i].file) });
    onProgress?.(i + 1, files.length);
  }
  return api<Upload>("upload", { files: payload });
}

export async function importPolicy(file: File) {
  return api<{ policy: string; checkpoints: string[] }>("import_policy", {
    name: file.name,
    data: await toBase64(file),
  });
}

export const shortPath = (path: string) => {
  const parts = path.split("/");
  return parts.slice(-2).join("/");
};
