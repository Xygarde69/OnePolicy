"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { OnePolicy } from "./brand";
import { api, type State } from "@/lib/api";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/train/", label: "Train", color: "bg-pink" },
  { href: "/onepolicy/", label: <OnePolicy className="text-base" />, color: "bg-cyan" },
];

export function SiteHeader() {
  const path = usePathname();
  const [online, setOnline] = useState<boolean | null>(null);
  useEffect(() => {
    api<State>("state").then(() => setOnline(true), () => setOnline(false));
  }, []);

  return (
    <header className="sticky top-0 z-40 border-b-3 border-ink bg-paper">
      <div className="mx-auto flex h-16 max-w-7xl items-center gap-4 px-4">
        <Link href="/" className="brutal brutal-press flex items-center gap-2 bg-yellow px-3 py-1">
          <OnePolicy className="text-2xl leading-none" />
        </Link>
        <nav className="flex gap-2">
          {NAV.map((n) => (
            <Link
              key={n.href}
              href={n.href}
              className={cn(
                "rounded-md border-3 border-ink px-3 py-1 text-sm font-bold uppercase",
                path?.startsWith(n.href) ? `${n.color} shadow-brutal-sm` : "bg-white hover:bg-muted",
              )}
            >
              {n.label}
            </Link>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-2 font-mono text-xs font-bold uppercase">
          <span
            className={cn(
              "size-3 rounded-full border-2 border-ink",
              online === null ? "bg-muted" : online ? "animate-pulse bg-lime" : "bg-red",
            )}
          />
          <span className="hidden sm:inline">
            {online === null ? "connecting" : online ? "server.py online" : "server.py offline"}
          </span>
        </div>
      </div>
    </header>
  );
}
