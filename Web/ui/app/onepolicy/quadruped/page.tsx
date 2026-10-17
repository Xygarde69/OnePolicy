"use client";

import { Boxes, SlidersHorizontal, Upload } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { PageTitle } from "@/components/brutal/page-title";
import { StepRail } from "@/components/brutal/step-rail";
import { AdapterReport, type UploadRef } from "@/components/flow/adapter-report";
import { ConventionDoc } from "@/components/flow/convention-doc";
import { FolderUpload } from "@/components/flow/folder-upload";
import { type Generated, GeneratorPanel } from "@/components/flow/generator-panel";
import { PartsPopup, useGeometry } from "@/components/flow/parts-popup";
import { PreloadedGrid } from "@/components/flow/preloaded-grid";
import { SimView } from "@/components/flow/sim-view";
import { TerrainPicker } from "@/components/flow/terrain-picker";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { api, type Inspect, type PlayStatus, type RobotSource, type State, shortPath } from "@/lib/api";

const STEPS = ["Robot", "Meet it", "Terrain", "Walk"];

export default function OnePolicyQuadruped() {
  const [step, setStep] = useState(0);
  const [state, setState] = useState<State | null>(null);
  const [tab, setTab] = useState("preloaded");
  const [robot, setRobot] = useState<string | null>(null);
  const [generated, setGenerated] = useState<Generated>({ seed: 42, params: {} });
  const [upload, setUpload] = useState<UploadRef | null>(null);
  const [inspect, setInspect] = useState<Inspect | null>(null);
  const [parts, setParts] = useState(false);
  const [terrain, setTerrain] = useState("flat");
  const [policy, setPolicy] = useState<string | null>(null);

  useEffect(() => {
    api<State>("state").then((s) => {
      setState(s);
      // ?policy=<path> (from the training page's "Walk it").
      const asked = new URLSearchParams(window.location.search).get("policy");
      setPolicy(asked && s.checkpoints.includes(asked) ? asked : s.onepolicy ?? s.checkpoints[0] ?? null);
    }, (e) => toast.error(e.message));
  }, []);

  const source: RobotSource | null = useMemo(() => {
    if (tab === "preloaded") return robot ? { source: "robot", robot } : null;
    if (tab === "generate") return { source: "random", seed: generated.seed, params: generated.params };
    if (upload && inspect && inspect.errors.length === 0)
      return { source: "upload", upload: upload.upload, xml: upload.xml, forward: upload.forward, xknee: upload.xknee ?? undefined };
    return null;
  }, [tab, robot, generated, upload, inspect]);

  const { geometry, error } = useGeometry(step >= 1 ? source : null);
  useEffect(() => {
    if (error) toast.error(error);
  }, [error]);

  const label = source?.source === "robot" ? source.robot : source?.source === "random" ? `generated #${source.seed}` : upload?.xml ?? "";

  const start = useCallback(
    () => api<PlayStatus>("play", { ...source, policy, terrain, drive: true }),
    [source, policy, terrain],
  );

  const meet = () => {
    setStep(1);
    setParts(true);
  };

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-col gap-8 px-4 py-10">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <PageTitle kicker="onepolicy ;) · quadruped" title="Walk any quadruped" color="bg-cyan" />
        {state && (
          <div className="flex flex-col gap-1">
            <span className="font-mono text-[11px] font-bold uppercase">Policy</span>
            <Select value={policy ?? ""} onValueChange={setPolicy}>
              <SelectTrigger className="w-72"><SelectValue placeholder="no checkpoint found" /></SelectTrigger>
              <SelectContent>
                {state.checkpoints.map((c) => (
                  <SelectItem key={c} value={c}>{c === state.onepolicy ? "OnePolicy.pt ;)" : shortPath(c)}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}
      </div>
      <StepRail steps={STEPS} current={step} onPick={setStep} />

      <AnimatePresence mode="wait">
        {step === 0 && (
          <motion.section key="robot" initial={{ opacity: 0, x: 40 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -40 }} className="flex flex-col gap-5">
            <Tabs value={tab} onValueChange={setTab} className="gap-5">
              <TabsList>
                <TabsTrigger value="preloaded"><Boxes /> Preloaded</TabsTrigger>
                <TabsTrigger value="generate"><SlidersHorizontal /> Generate your own</TabsTrigger>
                <TabsTrigger value="upload"><Upload /> Upload MJCF</TabsTrigger>
              </TabsList>
              <TabsContent value="preloaded" className="flex flex-col gap-5">
                {state && <PreloadedGrid robots={state.robots} selected={robot} onSelect={setRobot} />}
              </TabsContent>
              <TabsContent value="generate">
                <GeneratorPanel value={generated} onChange={setGenerated} />
              </TabsContent>
              <TabsContent value="upload" className="grid gap-6 lg:grid-cols-[1.5fr_1fr]">
                <div className="flex flex-col gap-5">
                  <FolderUpload
                    accent="bg-cyan"
                    onUploaded={(up, xml) => {
                      setInspect(null);
                      setUpload({ upload: up.upload, xml, forward: "auto", xknee: null });
                    }}
                  />
                  {upload && (
                    <AdapterReport
                      target={upload}
                      onResult={(r, t) => {
                        setInspect(r);
                        setUpload(t);
                      }}
                    />
                  )}
                </div>
                <ConventionDoc />
              </TabsContent>
            </Tabs>
            <div className="sticky bottom-4 z-10 flex justify-end">
              <Button size="lg" disabled={!source} onClick={meet}>
                {source ? `Use ${label} →` : tab === "upload" ? "Upload a robot that passes the check" : "Pick a robot"}
              </Button>
            </div>
          </motion.section>
        )}

        {step === 1 && (
          <motion.section key="meet" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col gap-4">
            <div className="brutal flex flex-wrap items-center gap-4 bg-white p-5">
              <span className="font-heading text-2xl uppercase">{geometry?.name ?? label}</span>
              <Button variant="outline" onClick={() => setParts(true)}>Show the parts</Button>
              <Button onClick={() => setStep(2)} disabled={!geometry}>Next: terrain →</Button>
            </div>
            {geometry?.fixes && geometry.fixes.length > 1 && (
              <details className="brutal bg-white">
                <summary className="cursor-pointer p-3 text-sm font-bold uppercase">What the fixer changed ({geometry.fixes.length})</summary>
                <ul className="max-h-60 overflow-auto border-t-3 border-ink p-3 font-mono text-xs">
                  {geometry.fixes.map((l, i) => <li key={i}>{l}</li>)}
                </ul>
              </details>
            )}
          </motion.section>
        )}

        {step === 2 && (
          <motion.section key="terrain" initial={{ opacity: 0, x: 40 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -40 }} className="flex flex-col gap-5">
            <h2 className="font-heading text-3xl uppercase">Where should it walk?</h2>
            <TerrainPicker value={terrain} available={state?.terrains ?? ["flat"]} onChange={setTerrain} />
            <Button size="lg" className="w-fit" onClick={() => setStep(3)}>Place it in the sim →</Button>
          </motion.section>
        )}

        {step === 3 && source && (
          <motion.section key="walk" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
            <SimView start={start} label={label} />
          </motion.section>
        )}
      </AnimatePresence>

      <PartsPopup
        open={parts}
        onOpenChange={setParts}
        geometry={geometry}
        onContinue={() => {
          setParts(false);
          setStep(2);
        }}
      />
    </div>
  );
}
