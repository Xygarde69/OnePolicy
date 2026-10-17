"use client";

import { FlaskConical } from "lucide-react";
import { motion } from "motion/react";
import { useRef, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { api, type ConditionTest } from "@/lib/api";
import { cn } from "@/lib/utils";

const EXAMPLES = [
  "leg_length > 0.3 and mass < 40",
  "abs(com_x) < 0.02",
  "thigh_length / calf_length > 1.1",
  "kp_calf < 200 and torque_calf < 150",
  "x_knee == 1",
];

const GROUPS: [string, (v: string) => boolean][] = [
  ["Size & mass", (v) => /mass|^s_|stand/.test(v)],
  ["Link lengths", (v) => /length|hip_link|base_|foot/.test(v)],
  ["Centre of mass", (v) => v.startsWith("com_")],
  ["Actuators", (v) => /^(kp|kd|torque)|velocity/.test(v)],
  ["Layout", (v) => v === "x_knee"],
];

// Each variable sits in the first group that matches it.
const groupOf = (v: string) => {
  const i = GROUPS.findIndex(([, match]) => match(v));
  return i < 0 ? GROUPS.length : i;
};

/** A morphology filter for the generated robots that train next to yours. */
export function ConditionEditor({ value, onChange, variables, functions }: {
  value: string;
  onChange: (v: string) => void;
  variables: Record<string, string>;
  functions: string[];
}) {
  const box = useRef<HTMLTextAreaElement>(null);
  const [test, setTest] = useState<ConditionTest | null>(null);
  const [busy, setBusy] = useState(false);

  const insert = (word: string) => {
    const el = box.current;
    const start = el?.selectionStart ?? value.length;
    const end = el?.selectionEnd ?? value.length;
    const next = value.slice(0, start) + word + value.slice(end);
    onChange(next);
    requestAnimationFrame(() => {
      el?.focus();
      el?.setSelectionRange(start + word.length, start + word.length);
    });
  };

  const run = async () => {
    setBusy(true);
    try {
      setTest(await api<ConditionTest>("condition_test", { condition: value, samples: 32 }));
    } catch (e) {
      setTest(null);
      toast.error((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const vars = Object.keys(variables);
  return (
    <div className="grid gap-6 lg:grid-cols-[1.3fr_1fr]">
      <div className="flex flex-col gap-4">
        <div className="brutal flex flex-col gap-3 bg-ink p-4 text-paper">
          <span className="font-mono text-xs font-bold uppercase text-yellow">keep generated robots where…</span>
          <Textarea
            ref={box}
            value={value}
            onChange={(e) => {
              onChange(e.target.value);
              setTest(null);
            }}
            spellCheck={false}
            placeholder="leg_length > 0.3 and mass < 40"
            className="min-h-28 border-paper bg-ink font-mono text-base text-lime shadow-none placeholder:text-paper/30 focus-visible:bg-ink"
          />
          <div className="flex flex-wrap items-center gap-2">
            <Button variant="lime" size="sm" disabled={busy} onClick={run}>
              <FlaskConical /> Test on 32 robots
            </Button>
            <span className="font-mono text-[11px] text-paper/60">
              python-like: and / or / not, + - * / **, comparisons, {functions.join("() ")}()
            </span>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              type="button"
              onClick={() => {
                onChange(ex);
                setTest(null);
              }}
              className="rounded-sm border-2 border-ink bg-white px-2 py-1 font-mono text-xs hover:bg-yellow"
            >
              {ex}
            </button>
          ))}
        </div>
        {test && (
          <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="brutal flex flex-col gap-3 bg-white p-4">
            <div className="flex items-baseline justify-between">
              <span className="font-heading text-xl uppercase">
                {test.kept}/{test.samples} robots kept
              </span>
              <span className="font-mono text-xs">
                {test.kept === 0 ? "nothing passes: training would fail" : `≈ ${Math.round((100 * test.kept) / test.samples)}% acceptance`}
              </span>
            </div>
            <div className="grid grid-cols-8 gap-1.5 sm:grid-cols-16">
              {test.rows.map((r, i) => (
                <Tooltip key={r.seed}>
                  <TooltipTrigger asChild>
                    <motion.span
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ delay: i * 0.02 }}
                      className={cn("aspect-square rounded-sm border-2 border-ink", r.ok ? "bg-lime" : "bg-white")}
                    />
                  </TooltipTrigger>
                  <TooltipContent>
                    seed {r.seed}: {r.mass.toFixed(1)} kg, leg {(r.leg_length * 100).toFixed(0)} cm · {r.ok ? "kept" : "skipped"}
                  </TooltipContent>
                </Tooltip>
              ))}
            </div>
          </motion.div>
        )}
      </div>
      <div className="brutal flex max-h-[32rem] flex-col gap-4 overflow-auto bg-white p-4">
        <span className="font-heading text-lg uppercase">Variables · click to insert</span>
        {[...GROUPS.map(([g]) => g), "Other"].map((group, gi) => vars.some((v) => groupOf(v) === gi) && (
          <div key={group} className="flex flex-col gap-2">
            <span className="font-mono text-[11px] font-bold uppercase text-ink/60">{group}</span>
            <div className="flex flex-wrap gap-1.5">
              {vars.filter((v) => groupOf(v) === gi).map((v) => (
                <Tooltip key={v}>
                  <TooltipTrigger asChild>
                    <button
                      type="button"
                      onClick={() => insert(v)}
                      className="rounded-sm border-2 border-ink bg-cyan px-1.5 py-0.5 font-mono text-xs font-bold hover:-translate-y-0.5 hover:shadow-brutal-sm"
                    >
                      {v}
                    </button>
                  </TooltipTrigger>
                  <TooltipContent>{variables[v]}</TooltipContent>
                </Tooltip>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
