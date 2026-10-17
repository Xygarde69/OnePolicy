"use client";

import { motion } from "motion/react";
import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

type Props = {
  index: number;
  title: React.ReactNode;
  subtitle: string;
  icon: LucideIcon;
  color: string;
  onClick: () => void;
  tag?: string;
  big?: boolean;
};

export function ChoiceCard({ index, title, subtitle, icon: Icon, color, onClick, tag, big }: Props) {
  return (
    <motion.button
      type="button"
      onClick={onClick}
      initial={{ opacity: 0, y: 24, rotate: index % 2 ? 1.5 : -1.5 }}
      animate={{ opacity: 1, y: 0, rotate: 0 }}
      transition={{ delay: 0.08 * index, type: "spring", stiffness: 260, damping: 20 }}
      className={cn(
        "brutal-lg brutal-press group relative flex w-full flex-col items-start gap-4 overflow-hidden text-left",
        big ? "min-h-80 p-8" : "min-h-56 p-6",
        color,
      )}
    >
      <span className="absolute top-4 right-4 font-heading text-6xl leading-none text-ink/15 select-none">
        {String(index + 1).padStart(2, "0")}
      </span>
      <span className="brutal flex size-14 items-center justify-center bg-white transition-transform group-hover:-rotate-6">
        <Icon className="size-7" strokeWidth={2.5} />
      </span>
      <span className={cn("font-heading leading-[0.95] uppercase", big ? "text-5xl" : "text-3xl")}>{title}</span>
      <span className="max-w-md text-base font-medium">{subtitle}</span>
      {tag && (
        <span className="mt-auto rounded-sm border-2 border-ink bg-white px-2 py-0.5 font-mono text-xs font-bold uppercase">
          {tag}
        </span>
      )}
      <span className="absolute right-6 bottom-6 font-heading text-3xl transition-transform group-hover:translate-x-2">
        →
      </span>
    </motion.button>
  );
}
