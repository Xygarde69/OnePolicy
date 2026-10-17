"use client";

import { Construction } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";

export function UnderConstruction({ open, onOpenChange, what }: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  what: string;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="overflow-hidden p-0 sm:max-w-lg" showCloseButton={false}>
        <div className="hazard animate-stripes h-6 border-b-3 border-ink" />
        <div className="flex flex-col items-center gap-4 px-6 py-6 text-center">
          <div className="brutal animate-bob flex size-20 items-center justify-center bg-yellow">
            <Construction className="size-10" strokeWidth={2.5} />
          </div>
          <DialogTitle className="font-heading text-3xl uppercase">This page is under construction</DialogTitle>
          <DialogDescription className="text-base text-ink">
            <b>{what}</b> isn&apos;t built yet. OnePolicy covers 12-DoF quadrupeds of any size for now;
            other body plans come later.
          </DialogDescription>
          <Button onClick={() => onOpenChange(false)}>Got it, back to quadrupeds</Button>
        </div>
        <div className="hazard animate-stripes h-6 border-t-3 border-ink" />
      </DialogContent>
    </Dialog>
  );
}

/** Inline "[under construction]" tag for options that don't work yet. */
export function ConstructionTag() {
  return (
    <span className="inline-flex items-center gap-1 rounded-sm border-2 border-ink bg-yellow px-1.5 py-0.5 font-mono text-[10px] font-bold uppercase">
      [under construction]
    </span>
  );
}
