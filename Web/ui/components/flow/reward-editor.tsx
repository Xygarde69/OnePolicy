"use client";

import { RotateCcw } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { RewardTerm } from "@/lib/api";
import { cn } from "@/lib/utils";

/** Edits to reward terms: {term: {weight?, params?: {name: value}}}. */
export type RewardEdits = Record<string, { weight?: number; params?: Record<string, number> }>;

// One iteration = 24 env steps (the curriculum counts env steps).
const STEPS_PER_ITERATION = 24;

function NumberCell({ value, original, onChange, disabled }: {
  value: number;
  original: number;
  onChange: (v: number) => void;
  disabled?: boolean;
}) {
  return (
    <Input
      type="number"
      step="any"
      value={Number.isFinite(value) ? value : ""}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value === "" ? original : Number(e.target.value))}
      className={cn("h-8 w-28 font-mono text-xs", value !== original && "bg-yellow")}
    />
  );
}

export function RewardEditor({ terms, edits, onChange }: {
  terms: RewardTerm[];
  edits: RewardEdits;
  onChange: (edits: RewardEdits) => void;
}) {
  const set = (name: string, patch: { weight?: number; params?: Record<string, number> }) => {
    const prev = edits[name] ?? {};
    onChange({ ...edits, [name]: { ...prev, ...patch, params: { ...prev.params, ...patch.params } } });
  };
  const changed = Object.entries(edits).filter(([name, e]) => {
    const t = terms.find((t) => t.name === name)!;
    return (e.weight !== undefined && e.weight !== t.weight) ||
      Object.entries(e.params ?? {}).some(([k, v]) => v !== t.params[k]);
  }).length;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-3">
        <Badge className="bg-lime">{terms.filter((t) => t.weight > 0).length} bonuses</Badge>
        <Badge variant="secondary">{terms.filter((t) => t.weight < 0).length} penalties</Badge>
        <Badge variant="outline">{changed} changed</Badge>
        <Button size="sm" variant="outline" className="ml-auto" disabled={!changed} onClick={() => onChange({})}>
          <RotateCcw /> Reset all
        </Button>
      </div>
      <div className="brutal overflow-auto bg-white">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Term</TableHead>
              <TableHead>What it does</TableHead>
              <TableHead>Weight</TableHead>
              <TableHead>Parameters</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {terms.map((t) => {
              const e = edits[t.name] ?? {};
              const weight = e.weight ?? t.weight;
              return (
                <TableRow key={t.name} className={cn(e.weight !== undefined && e.weight !== t.weight && "bg-yellow/40")}>
                  <TableCell className="align-top">
                    <div className="flex items-center gap-2">
                      <span className={cn("size-3 shrink-0 rounded-sm border-2 border-ink", t.weight > 0 ? "bg-lime" : "bg-pink")} />
                      <span className="font-mono text-xs font-bold">{t.name}</span>
                    </div>
                  </TableCell>
                  <TableCell className="max-w-sm align-top text-xs whitespace-normal">{t.doc || t.func}</TableCell>
                  <TableCell className="align-top">
                    {t.curriculum ? (
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <span className="inline-flex flex-col gap-0.5 rounded-sm border-2 border-ink bg-cyan px-2 py-1 font-mono text-[11px] font-bold">
                            curriculum
                            <span>
                              {t.curriculum.map((s) => s.weight).join(" → ")} @ it{" "}
                              {t.curriculum.slice(1).map((s) => s.step / STEPS_PER_ITERATION).join(", ")}
                            </span>
                          </span>
                        </TooltipTrigger>
                        <TooltipContent>The command curriculum sets this weight; it can&apos;t be overridden.</TooltipContent>
                      </Tooltip>
                    ) : (
                      <NumberCell value={weight} original={t.weight} onChange={(v) => set(t.name, { weight: v })} />
                    )}
                  </TableCell>
                  <TableCell className="align-top">
                    <div className="flex flex-col gap-1">
                      {Object.entries(t.params).map(([k, v]) => (
                        <label key={k} className="flex items-center gap-2 font-mono text-[11px]">
                          <span className="w-32 truncate">{k}</span>
                          <NumberCell
                            value={e.params?.[k] ?? v}
                            original={v}
                            onChange={(nv) => set(t.name, { params: { [k]: nv } })}
                          />
                        </label>
                      ))}
                      {Object.keys(t.params).length === 0 && <span className="text-xs text-ink/40">—</span>}
                    </div>
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
