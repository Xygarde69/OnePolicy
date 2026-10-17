"use client";

import { AlertTriangle, CheckCircle2, ChevronDown, Save, Wrench } from "lucide-react";
import { motion } from "motion/react";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Textarea } from "@/components/ui/textarea";
import { api, type Inspect } from "@/lib/api";
import { cn } from "@/lib/utils";

export type UploadRef = { upload: string; xml: string; forward: string; xknee: boolean | null };

/** Runs the convention check on an uploaded MJCF and lets the user fix it in place. */
export function AdapterReport({ target, onResult }: {
  target: UploadRef;
  onResult: (inspect: Inspect, target: UploadRef) => void;
}) {
  const [inspect, setInspect] = useState<Inspect | null>(null);
  const [busy, setBusy] = useState(false);
  const [text, setText] = useState("");
  const [forward, setForward] = useState(target.forward);
  const [xknee, setXknee] = useState<boolean | null>(target.xknee);
  const [showTable, setShowTable] = useState(false);

  const body = (extra: object = {}) => ({
    kind: "upload", upload: target.upload, xml: target.xml, forward, xknee, ...extra,
  });

  const run = async (endpoint: "inspect" | "save", extra: object = {}) => {
    setBusy(true);
    try {
      const r = await api<Inspect>(endpoint, body(extra));
      setInspect(r);
      setText(r.text ?? "");
      if (xknee === null) setXknee(r.xknee);
      setShowTable(r.errors.length > 0);
      onResult(r, { ...target, forward, xknee: xknee ?? r.xknee });
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    run("inspect");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target.upload, target.xml]);

  if (!inspect) {
    return (
      <div className="brutal flex items-center gap-3 bg-white p-6 font-heading text-xl uppercase">
        <span className="size-5 animate-spin rounded-sm border-3 border-ink border-t-pink" />
        Checking {target.xml} against the convention…
      </div>
    );
  }

  const ok = inspect.errors.length === 0;
  return (
    <div className="flex flex-col gap-4">
      <motion.div
        key={String(ok) + inspect.errors.join()}
        initial={{ scale: 0.96, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        className={cn("brutal flex flex-col gap-3 p-5", ok ? "bg-lime" : "bg-red")}
      >
        <div className="flex items-center gap-3">
          {ok ? <CheckCircle2 className="size-8" strokeWidth={2.5} /> : <AlertTriangle className="size-8" strokeWidth={2.5} />}
          <div className="font-heading text-2xl uppercase">
            {ok ? `${inspect.file} passes the convention` : `${inspect.errors.length} adapter error${inspect.errors.length > 1 ? "s" : ""}`}
          </div>
        </div>
        {inspect.errors.map((err, i) => {
          const [msg, ...rest] = err.split("\n");
          return (
            <div key={i} className="rounded-md border-3 border-ink bg-white p-3 text-sm">
              <div className="font-bold">{msg}</div>
              {rest.length > 0 && (
                <pre className="mt-2 font-mono text-xs whitespace-pre-wrap">
                  {rest.map((l) => (l.trim().startsWith("FIX:") ? `🔧 ${l.trim()}` : l)).join("\n")}
                </pre>
              )}
            </div>
          );
        })}
      </motion.div>

      {inspect.report.length > 0 && (
        <details className="brutal bg-white" open={!ok}>
          <summary className="flex cursor-pointer items-center gap-2 p-3 text-sm font-bold uppercase">
            <Wrench className="size-4" /> What the fixer and adapter did ({inspect.report.length})
            <ChevronDown className="ml-auto size-4" />
          </summary>
          <ul className="max-h-60 overflow-auto border-t-3 border-ink p-3 font-mono text-xs">
            {inspect.report.map((line, i) => (
              <li key={i} className={cn("py-0.5", line.startsWith("WARN") && "font-bold text-red")}>
                {line}
              </li>
            ))}
          </ul>
        </details>
      )}

      {inspect.table && (
        <div className="brutal overflow-hidden bg-white">
          <button
            type="button"
            onClick={() => setShowTable((s) => !s)}
            className="flex w-full items-center gap-2 p-3 text-left text-sm font-bold uppercase"
          >
            Joint table · {inspect.table.joints} joints · {inspect.table.legs} legs · {inspect.table.mass.toFixed(2)} kg
            <ChevronDown className={cn("ml-auto size-4 transition-transform", showTable && "rotate-180")} />
          </button>
          {showTable && (
            <div className="max-h-96 overflow-auto border-t-3 border-ink">
              <Table>
                <TableHeader>
                  <TableRow>
                    {["#", "joint", "type", "body", "axis (base)", "range", "adapter's guess"].map((h) => (
                      <TableHead key={h}>{h}</TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody className="font-mono text-xs">
                  {inspect.table.rows.map((r) => (
                    <TableRow key={r.index} className={cn(!r.ok && "bg-red/40")}>
                      <TableCell>{r.index}</TableCell>
                      <TableCell className="font-bold">{r.name}</TableCell>
                      <TableCell>{r.type}</TableCell>
                      <TableCell>{r.body}</TableCell>
                      <TableCell>{r.axis}</TableCell>
                      <TableCell>{r.range}</TableCell>
                      <TableCell>{r.guess}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </div>
      )}

      <div className="brutal flex flex-wrap items-end gap-6 bg-white p-4">
        <div className="flex flex-col gap-2">
          <Label className="text-xs font-bold uppercase">Forward axis</Label>
          <Select value={forward} onValueChange={setForward}>
            <SelectTrigger className="w-32"><SelectValue /></SelectTrigger>
            <SelectContent>
              {["auto", "+x", "-x", "+y", "-y"].map((f) => <SelectItem key={f} value={f}>{f}</SelectItem>)}
            </SelectContent>
          </Select>
        </div>
        <label className="flex items-center gap-3 text-sm font-bold">
          <Switch checked={!!xknee} onCheckedChange={setXknee} />
          Rear knees bend forward (&gt;&lt;, ANYmal)
        </label>
        <Button variant="cyan" disabled={busy} onClick={() => run("inspect")}>Re-check</Button>
      </div>

      {!ok && inspect.text !== undefined && (
        <div className="brutal flex flex-col gap-3 bg-white p-4">
          <div className="flex items-center justify-between">
            <span className="font-heading text-lg uppercase">Fix it here: {inspect.file}</span>
            <span className="font-mono text-xs">the original is kept as {inspect.file}.orig</span>
          </div>
          <Textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            spellCheck={false}
            className="h-80 font-mono text-xs [field-sizing:fixed]"
          />
          <Button className="w-fit" disabled={busy} onClick={() => run("save", { text })}>
            <Save /> Save & re-check
          </Button>
        </div>
      )}
    </div>
  );
}
