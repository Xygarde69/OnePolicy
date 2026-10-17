"use client";

import { FileCode2, FileBox, FileQuestion, Folder, FolderOpen, Upload as UploadIcon } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useCallback, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { type PickedFile, type Upload, uploadFiles } from "@/lib/api";
import { cn } from "@/lib/utils";

type Node = { name: string; path: string; dir: boolean; children: Node[]; size: number };

function buildTree(files: PickedFile[]): Node {
  const root: Node = { name: "", path: "", dir: true, children: [], size: 0 };
  for (const f of files) {
    let node = root;
    const parts = f.path.split("/");
    parts.forEach((part, i) => {
      const leaf = i === parts.length - 1;
      let child = node.children.find((c) => c.name === part);
      if (!child) {
        child = { name: part, path: parts.slice(0, i + 1).join("/"), dir: !leaf, children: [], size: 0 };
        node.children.push(child);
      }
      child.size += f.file.size;
      node = child;
    });
  }
  const sort = (n: Node) => {
    n.children.sort((a, b) => Number(b.dir) - Number(a.dir) || a.name.localeCompare(b.name));
    n.children.forEach(sort);
  };
  sort(root);
  return root;
}

const MESH = /\.(stl|obj|msh|dae|ply)$/i;

function flatten(node: Node, depth = 0, out: { node: Node; depth: number }[] = []) {
  for (const c of node.children) {
    out.push({ node: c, depth });
    if (c.dir) flatten(c, depth + 1, out);
  }
  return out;
}

// Directory drops: walk FileSystemEntry trees (Chrome, Firefox, Safari).
async function readEntry(entry: FileSystemEntry, prefix: string, out: PickedFile[]) {
  if (entry.isFile) {
    const file = await new Promise<File>((res, rej) => (entry as FileSystemFileEntry).file(res, rej));
    out.push({ path: prefix + entry.name, file });
  } else if (entry.isDirectory) {
    const reader = (entry as FileSystemDirectoryEntry).createReader();
    let batch: FileSystemEntry[];
    do {
      batch = await new Promise<FileSystemEntry[]>((res, rej) => reader.readEntries(res, rej));
      for (const e of batch) await readEntry(e, `${prefix}${entry.name}/`, out);
    } while (batch.length);
  }
}

const fmtSize = (n: number) => (n > 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1e3))} kB`);

export function FolderUpload({ onUploaded, accent = "bg-pink" }: {
  onUploaded: (upload: Upload, xml: string) => void;
  accent?: string;
}) {
  const [files, setFiles] = useState<PickedFile[]>([]);
  const [drag, setDrag] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [upload, setUpload] = useState<Upload | null>(null);
  const folderInput = useRef<HTMLInputElement>(null);
  const filesInput = useRef<HTMLInputElement>(null);

  const tree = useMemo(() => flatten(buildTree(files)), [files]);

  const pick = useCallback((list: PickedFile[]) => {
    setUpload(null);
    setProgress(null);
    setFiles(list.filter((f) => !f.path.split("/").some((p) => p.startsWith("."))));
  }, []);

  const onDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setDrag(false);
    const out: PickedFile[] = [];
    const entries = [...e.dataTransfer.items].map((i) => i.webkitGetAsEntry()).filter(Boolean) as FileSystemEntry[];
    for (const entry of entries) await readEntry(entry, "", out);
    pick(out);
  };

  const fromInput = (list: FileList | null) =>
    pick([...(list ?? [])].map((f) => ({ path: f.webkitRelativePath || f.name, file: f })));

  const send = async () => {
    try {
      setProgress(0);
      const up = await uploadFiles(files, (d, t) => setProgress((90 * d) / t));
      setProgress(100);
      setUpload(up);
      if (up.xmls.length === 1) onUploaded(up, up.xmls[0]);
    } catch (err) {
      setProgress(null);
      toast.error(String((err as Error).message));
    }
  };

  const xmlCount = files.filter((f) => f.path.toLowerCase().endsWith(".xml")).length;
  const meshCount = files.filter((f) => MESH.test(f.path)).length;

  return (
    <div className="flex flex-col gap-4">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={onDrop}
        className={cn(
          "brutal relative flex min-h-56 flex-col items-center justify-center gap-4 border-dashed p-8 text-center transition-colors",
          drag ? `${accent} scale-[1.01]` : "bg-white",
        )}
      >
        <motion.div
          animate={drag ? { rotate: [0, -8, 8, 0], scale: 1.15 } : { rotate: 0, scale: 1 }}
          transition={{ duration: 0.5 }}
          className={cn("brutal flex size-20 items-center justify-center", accent)}
        >
          {drag ? <FolderOpen className="size-10" strokeWidth={2.5} /> : <Folder className="size-10" strokeWidth={2.5} />}
        </motion.div>
        <div className="font-heading text-2xl uppercase">Drop your robot folder here</div>
        <p className="max-w-md text-sm font-medium">
          The MJCF (<code className="font-mono">.xml</code>) and every mesh it uses. A scene file that
          <code className="font-mono"> &lt;include&gt;</code>s the robot is fine: the robot file is picked.
        </p>
        <div className="flex flex-wrap justify-center gap-3">
          <Button variant="outline" onClick={() => folderInput.current?.click()}>
            <FolderOpen /> Choose folder
          </Button>
          <Button variant="outline" onClick={() => filesInput.current?.click()}>
            <FileCode2 /> Choose files
          </Button>
        </div>
        <input
          ref={folderInput}
          type="file"
          className="hidden"
          multiple
          {...({ webkitdirectory: "" } as Record<string, string>)}
          onChange={(e) => fromInput(e.target.files)}
        />
        <input ref={filesInput} type="file" className="hidden" multiple onChange={(e) => fromInput(e.target.files)} />
      </div>

      <AnimatePresence>
        {files.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="brutal overflow-hidden bg-ink text-paper"
          >
            <div className="flex items-center justify-between border-b-3 border-paper/20 px-4 py-2 font-mono text-xs font-bold uppercase">
              <span>
                {files.length} files · {xmlCount} xml · {meshCount} meshes
              </span>
              <span className="text-yellow">{fmtSize(files.reduce((a, f) => a + f.file.size, 0))}</span>
            </div>
            <div className="relative max-h-72 overflow-auto px-4 py-3 font-mono text-sm">
              {/* A scan line sweeping the tree while the files go up. */}
              {progress !== null && progress < 100 && (
                <motion.div
                  className="pointer-events-none absolute inset-x-0 h-8 bg-gradient-to-b from-transparent via-yellow/30 to-transparent"
                  animate={{ top: ["0%", "100%"] }}
                  transition={{ duration: 1.1, repeat: Infinity, ease: "linear" }}
                />
              )}
              {tree.map(({ node, depth }, i) => {
                const xml = node.name.toLowerCase().endsWith(".xml");
                const mesh = MESH.test(node.name);
                const Icon = node.dir ? FolderOpen : xml ? FileCode2 : mesh ? FileBox : FileQuestion;
                return (
                  <motion.div
                    key={node.path}
                    initial={{ opacity: 0, x: -16 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: Math.min(i * 0.025, 1.2), type: "spring", stiffness: 400, damping: 30 }}
                    className="flex items-center gap-2 py-0.5"
                    style={{ paddingLeft: depth * 18 }}
                  >
                    <span className="text-paper/30">{depth > 0 ? "└" : ""}</span>
                    <Icon
                      className={cn("size-4 shrink-0", node.dir ? "text-yellow" : xml ? "text-pink" : mesh ? "text-cyan" : "text-paper/50")}
                    />
                    <span className={cn(xml && "font-bold text-pink", node.dir && "font-bold")}>{node.name}</span>
                    {!node.dir && <span className="ml-auto text-xs text-paper/40">{fmtSize(node.size)}</span>}
                  </motion.div>
                );
              })}
            </div>
            <div className="flex flex-col gap-3 border-t-3 border-paper/20 p-4">
              {progress !== null && <Progress value={progress} />}
              {!upload && (
                <Button onClick={send} disabled={progress !== null || xmlCount === 0} className="w-fit">
                  <UploadIcon /> {xmlCount === 0 ? "No .xml in these files" : "Upload & check"}
                </Button>
              )}
              {upload && upload.xmls.length > 1 && (
                <div className="flex flex-col gap-2">
                  <span className="text-xs font-bold uppercase text-yellow">Several MJCFs: which one is the robot?</span>
                  <div className="flex flex-wrap gap-2">
                    {/* server.py lists the likely robot file first (scenes and MJX variants last). */}
                    {upload.xmls.map((x, i) => (
                      <Button key={x} size="sm" variant={i === 0 ? "default" : "outline"} onClick={() => onUploaded(upload, x)}>
                        {x}
                        {i === 0 && " ★"}
                      </Button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
