"use client";

import { Download, FileUp, Play } from "lucide-react";
import Link from "next/link";
import { useRef, useState } from "react";
import { toast } from "sonner";
import { OnePolicy } from "@/components/brutal/brand";
import { Button } from "@/components/ui/button";
import { downloadUrl, importPolicy, shortPath } from "@/lib/api";

/** The run's checkpoints to download, and a .pt import (to play or resume). */
export function PolicyFiles({ checkpoints, onImported }: {
  checkpoints: string[];
  onImported?: (path: string) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [imported, setImported] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const pick = async (file: File | undefined) => {
    if (!file) return;
    setBusy(true);
    try {
      const r = await importPolicy(file);
      setImported(r.policy);
      onImported?.(r.policy);
      toast.success(`Imported ${file.name}`);
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const pts = [...checkpoints].reverse();
  return (
    <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
      <div className="brutal flex flex-col bg-white">
        <div className="border-b-3 border-ink bg-yellow px-4 py-2 font-heading uppercase">Checkpoints of this run</div>
        {pts.length === 0 && <div className="p-4 text-sm">No checkpoint yet: one is saved every “checkpoint every” iterations.</div>}
        <ul className="max-h-80 divide-y-2 divide-ink overflow-auto">
          {pts.map((p, i) => (
            <li key={p} className="flex items-center gap-3 px-4 py-2">
              <span className="font-mono text-sm font-bold">{p.split("/").pop()}</span>
              {i === 0 && <span className="rounded-sm border-2 border-ink bg-lime px-1.5 font-mono text-[10px] font-bold uppercase">latest</span>}
              <div className="ml-auto flex gap-2">
                {p.endsWith(".pt") && (
                  <Button size="sm" variant="cyan" asChild>
                    <Link href={`/onepolicy/quadruped/?policy=${encodeURIComponent(p)}`}>
                      <Play /> Walk it
                    </Link>
                  </Button>
                )}
                <Button size="sm" variant="outline" asChild>
                  <a href={downloadUrl(p)} download>
                    <Download /> Download
                  </a>
                </Button>
              </div>
            </li>
          ))}
        </ul>
      </div>
      <div className="brutal flex flex-col gap-3 bg-violet p-5">
        <span className="font-heading text-xl uppercase">Import a .pt</span>
        <p className="text-sm font-medium">
          A checkpoint from another machine or an earlier run. It lands in its own run folder, so you can walk it in
          <OnePolicy /> or start the next training from it.
        </p>
        <Button className="w-fit" disabled={busy} onClick={() => input.current?.click()}>
          <FileUp /> Choose .pt file
        </Button>
        <input ref={input} type="file" accept=".pt" className="hidden" onChange={(e) => pick(e.target.files?.[0])} />
        {imported && <code className="rounded-sm border-2 border-ink bg-white p-2 font-mono text-xs break-all">{shortPath(imported)}</code>}
      </div>
    </div>
  );
}
