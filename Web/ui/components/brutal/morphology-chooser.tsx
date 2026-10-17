"use client";

import { PawPrint, Shapes } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { ChoiceCard } from "./choice-card";
import { PageTitle } from "./page-title";
import { UnderConstruction } from "./under-construction";

/** Step one of both flows: quadruped (works) or custom morphology (not yet). */
export function MorphologyChooser({ kicker, title, intro, next, color }: {
  kicker: React.ReactNode;
  title: React.ReactNode;
  intro: React.ReactNode;
  next: string;
  color: string;
}) {
  const router = useRouter();
  const [construction, setConstruction] = useState(false);
  return (
    <section className="mx-auto flex w-full max-w-7xl flex-col gap-10 px-4 py-12">
      <PageTitle kicker={kicker} title={title} color={color}>{intro}</PageTitle>
      <div className="grid gap-8 md:grid-cols-2">
        <ChoiceCard
          index={0}
          color="bg-lime"
          icon={PawPrint}
          title="Quadruped"
          subtitle="Four legs, three hinge joints each (abduction, hip, knee). Any size, >> or >< knees."
          tag="ready"
          onClick={() => router.push(next)}
        />
        <ChoiceCard
          index={1}
          color="bg-violet"
          icon={Shapes}
          title="Custom morphology"
          subtitle="Hexapods, bipeds, wheels-on-legs, whatever you can write in MJCF."
          tag="under construction"
          onClick={() => setConstruction(true)}
        />
      </div>
      <UnderConstruction open={construction} onOpenChange={setConstruction} what="Custom morphologies" />
    </section>
  );
}
