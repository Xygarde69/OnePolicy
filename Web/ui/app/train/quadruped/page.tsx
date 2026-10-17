"use client";

import { Rocket } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { PageTitle } from "@/components/brutal/page-title";
import { StepRail } from "@/components/brutal/step-rail";
import { AdapterReport, type UploadRef } from "@/components/flow/adapter-report";
import { ConditionEditor } from "@/components/flow/condition-editor";
import { FolderUpload } from "@/components/flow/folder-upload";
import { LimitRow, LimitsEditor } from "@/components/flow/limits-editor";
import { PartsPopup, useGeometry } from "@/components/flow/parts-popup";
import { PolicyFiles } from "@/components/flow/policy-files";
import { type RewardEdits, RewardEditor } from "@/components/flow/reward-editor";
import { TrainingView } from "@/components/flow/training-view";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { api, type Inspect, type RewardTable, type State, type TrainStatus, shortPath } from "@/lib/api";

const STEPS = ["Upload", "Check", "Rewards", "Limits", "Conditions", "Train", "Policy"];
const SCRATCH = "scratch";

function Panel({ title, children, hint }: { title: string; hint?: string; children: React.ReactNode }) {
  return (
    <motion.section
      initial={{ opacity: 0, x: 40 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -40 }}
      transition={{ type: "spring", stiffness: 300, damping: 30 }}
      className="flex flex-col gap-5"
    >
      <div className="flex flex-col gap-1">
        <h2 className="font-heading text-3xl uppercase">{title}</h2>
        {hint && <p className="max-w-3xl font-medium">{hint}</p>}
      </div>
      {children}
    </motion.section>
  );
}

export default function TrainQuadruped() {
  const [step, setStep] = useState(0);
  const [target, setTarget] = useState<UploadRef | null>(null);
  const [inspect, setInspect] = useState<Inspect | null>(null);
  const [parts, setParts] = useState(false);
  const [table, setTable] = useState<RewardTable | null>(null);
  const [state, setState] = useState<State | null>(null);
  const [edits, setEdits] = useState<RewardEdits>({});
  const [limits, setLimits] = useState<Record<string, number>>({});
  const [condition, setCondition] = useState("");
  const [name, setName] = useState("");
  const [overwrite, setOverwrite] = useState(false);
  const [fraction, setFraction] = useState(0.5);
  const [startFrom, setStartFrom] = useState(SCRATCH);
  const [status, setStatus] = useState<TrainStatus | null>(null);
  const [starting, setStarting] = useState(false);

  const ok = !!inspect && inspect.errors.length === 0;
  const { geometry } = useGeometry(
    ok && target ? { source: "upload", upload: target.upload, xml: target.xml, forward: target.forward, xknee: target.xknee ?? undefined } : null,
  );

  useEffect(() => {
    api<RewardTable>("rewards").then(setTable, (e) => toast.error(e.message));
    api<State>("state").then((s) => {
      setState(s);
      if (s.onepolicy) setStartFrom(s.onepolicy);
    });
    // A run already going (page reload): jump to it.
    api<TrainStatus>("train_status").then((s) => {
      setStatus(s);
      if (s.running) setStep(5);
    });
  }, []);

  const poll = useCallback(() => api<TrainStatus>("train_status").then(setStatus).catch(() => {}), []);
  useEffect(() => {
    if (step < 5 && !status?.running) return;
    const t = setInterval(poll, 2000);
    return () => clearInterval(t);
  }, [step, status?.running, poll]);

  const start = async () => {
    if (!target || !table) return;
    setStarting(true);
    try {
      const installed = await api<{ robot: string }>("install", {
        kind: "upload", upload: target.upload, xml: target.xml, forward: target.forward,
        xknee: target.xknee, name, overwrite,
      });
      setStatus(await api<TrainStatus>("train", {
        robot: installed.robot,
        fraction,
        rewards: edits,
        limits,
        condition,
        policy: startFrom === SCRATCH ? null : startFrom,
      }));
      toast.success(`Training ${installed.robot} on this machine`);
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setStarting(false);
    }
  };

  const stop = async () => setStatus(await api<TrainStatus>("train_stop", {}));
  const began = status && status.phase !== "idle";

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-col gap-8 px-4 py-10">
      <PageTitle kicker="train your own robot · quadruped" title="Train it yourself" color="bg-pink" />
      <StepRail steps={STEPS} current={step} onPick={setStep} />

      <AnimatePresence mode="wait">
        {step === 0 && (
          <Panel key="upload" title="1 · Upload the MJCF" hint="Your robot's folder: its XML and meshes. Nothing leaves this machine.">
            <FolderUpload
              onUploaded={(up, xml) => {
                setTarget({ upload: up.upload, xml, forward: "auto", xknee: null });
                setInspect(null);
                setStep(1);
              }}
            />
          </Panel>
        )}

        {step === 1 && target && (
          <Panel key="check" title="2 · Convention check" hint="The fixer and adapter rewrite your robot into the training convention. Errors come with a fix.">
            <AdapterReport
              target={target}
              onResult={(r, t) => {
                setInspect(r);
                setTarget(t);
                if (r.default_name && !name) setName(r.default_name);
                if (r.errors.length === 0) setParts(true);
              }}
            />
            {ok && (
              <div className="flex gap-3">
                <Button variant="outline" onClick={() => setParts(true)}>Show the parts again</Button>
                <Button onClick={() => setStep(2)}>Next: rewards →</Button>
              </div>
            )}
            <PartsPopup open={parts} onOpenChange={setParts} geometry={geometry} onContinue={() => { setParts(false); setStep(2); }} />
          </Panel>
        )}

        {step === 2 && table && (
          <Panel
            key="rewards"
            title="3 · Rewards"
            hint="Every term is in Go2 units or a ratio to the robot's own size, so it means the same on a small and a large robot. Edit weights and parameters; changed cells turn yellow."
          >
            <RewardEditor terms={table.terms} edits={edits} onChange={setEdits} />
            <Button className="w-fit" onClick={() => setStep(3)}>Next: limits →</Button>
          </Panel>
        )}

        {step === 3 && table && (
          <Panel key="limits" title="4 · Limits & parameters" hint="How big the run is, when an episode ends, and how varied the generated robots are.">
            <LimitsEditor limits={table.limits} values={limits} onChange={setLimits} />
            <Button className="w-fit" onClick={() => setStep(4)}>Next: conditions →</Button>
          </Panel>
        )}

        {step === 4 && table && (
          <Panel
            key="conditions"
            title="5 · Your own conditions"
            hint="Your robot trains next to generated ones. Keep only the generated robots you care about, using their COM, actuator gains, link lengths and more. Leave it empty to keep all."
          >
            <ConditionEditor value={condition} onChange={setCondition} variables={table.variables} functions={table.functions} />
            <Button className="w-fit" onClick={() => setStep(5)}>Next: train →</Button>
          </Panel>
        )}

        {step === 5 && (
          <Panel key="train" title="6 · Train" hint="Runs Training/train.py on this machine's GPU. Closing the page doesn't stop it.">
            {began && status ? (
              <>
                <TrainingView status={status} onStop={stop} />
                {!status.running && (
                  <div className="flex gap-3">
                    <Button onClick={() => setStep(6)}>Next: policy files →</Button>
                    <Button variant="outline" onClick={() => setStatus({ ...status, phase: "idle" })}>Set up another run</Button>
                  </div>
                )}
              </>
            ) : (
              <div className="grid gap-5 lg:grid-cols-2">
                <div className="brutal flex flex-col gap-4 bg-white p-5">
                  <div className="flex flex-col gap-2">
                    <Label className="text-xs font-bold uppercase">Robot name (installed into quadrupeds/)</Label>
                    <Input value={name} onChange={(e) => setName(e.target.value)} className="font-mono" />
                    {state?.robots.includes(name) && (
                      <label className="flex items-center gap-3 text-sm font-bold">
                        <Switch checked={overwrite} onCheckedChange={setOverwrite} /> “{name}” exists: overwrite it
                      </label>
                    )}
                  </div>
                  <LimitRow
                    limit={{ label: "Share of envs on your robot", min: 0.05, max: 0.95, step: 0.05, unit: "", value: 0.5 }}
                    value={fraction}
                    onChange={setFraction}
                  />
                  <div className="flex flex-col gap-2">
                    <Label className="text-xs font-bold uppercase">Start from</Label>
                    <Select value={startFrom} onValueChange={setStartFrom}>
                      <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value={SCRATCH}>Scratch (random weights)</SelectItem>
                        {state?.checkpoints.map((c) => (
                          <SelectItem key={c} value={c}>
                            {c === state.onepolicy ? "OnePolicy.pt (fine-tune it)" : shortPath(c)}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                <div className="brutal flex flex-col gap-3 bg-yellow p-5">
                  <span className="font-heading text-2xl uppercase">Ready?</span>
                  <ul className="flex flex-col gap-1 font-mono text-sm">
                    <li>robot: {target?.xml ?? "— upload one first"}</li>
                    <li>{Math.round(fraction * 100)}% of {limits.num_envs ?? table?.limits.find((l) => l.key === "num_envs")?.value} envs on it</li>
                    <li>{Object.keys(edits).length} reward terms edited</li>
                    <li>{Object.keys(limits).length} limits changed</li>
                    <li>condition: {condition.trim() || "none"}</li>
                  </ul>
                  <Button size="lg" variant="ink" className="mt-auto w-fit" disabled={!ok || !name || starting} onClick={start}>
                    <Rocket /> {starting ? "Starting…" : "Start training"}
                  </Button>
                  {!ok && <span className="text-sm font-bold">Upload a robot that passes the check first.</span>}
                </div>
              </div>
            )}
          </Panel>
        )}

        {step === 6 && (
          <Panel key="policy" title="7 · Your policy" hint="Download the checkpoints (.pt for rsl_rl, policy.onnx for deployment), or import one.">
            <PolicyFiles checkpoints={status?.checkpoints ?? []} />
          </Panel>
        )}
      </AnimatePresence>
    </div>
  );
}
