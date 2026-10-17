"use client";

import { Check } from "lucide-react";
import { cn } from "@/lib/utils";

export function StepRail({ steps, current, onPick }: {
  steps: string[];
  current: number;
  /** Lets the user jump back to a finished step. */
  onPick?: (i: number) => void;
}) {
  return (
    <ol className="flex flex-wrap gap-2">
      {steps.map((s, i) => {
        const done = i < current;
        return (
          <li key={s}>
            <button
              type="button"
              disabled={!done || !onPick}
              onClick={() => onPick?.(i)}
              className={cn(
                "flex items-center gap-2 rounded-md border-3 border-ink px-3 py-1.5 text-xs font-bold uppercase",
                i === current && "bg-yellow shadow-brutal",
                done && "bg-lime hover:-translate-y-0.5",
                i > current && "bg-white text-ink/50",
              )}
            >
              <span className="flex size-5 items-center justify-center rounded-sm border-2 border-ink bg-white font-mono text-[11px] text-ink">
                {done ? <Check className="size-3" strokeWidth={3} /> : i + 1}
              </span>
              {s}
            </button>
          </li>
        );
      })}
    </ol>
  );
}
