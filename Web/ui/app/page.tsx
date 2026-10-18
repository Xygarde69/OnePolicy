"use client";

import { Bot, Dumbbell } from "lucide-react";
import { useRouter } from "next/navigation";
import { ChoiceCard } from "@/components/brutal/choice-card";
import { OnePolicy } from "@/components/brutal/brand";
import { Marquee } from "@/components/brutal/marquee";
import { HeroRobot } from "@/components/three/hero-robot";

export default function Home() {
  const router = useRouter();
  return (
    <>
      <section className="mx-auto grid w-full max-w-7xl gap-10 px-4 py-12 lg:grid-cols-[1.2fr_1fr] lg:py-16">
        <div className="flex flex-col justify-center gap-6">
          <span className="w-fit -rotate-2 rounded-sm border-3 border-ink bg-pink px-3 py-1 font-mono text-sm font-bold uppercase shadow-brutal">
            one network · thousands of bodies
          </span>
          <h1 className="font-heading text-6xl leading-[0.9] uppercase sm:text-8xl">
            One policy.
            <br />
            <span className="relative inline-block">
              <span className="absolute inset-x-0 bottom-1 -z-10 h-1/3 bg-yellow" />
              Any
            </span>{" "}
            quadruped.
          </h1>
          <p className="max-w-xl text-xl font-medium">
            Trained on thousands of procedurally generated quadrupeds, 1/4× to 4× a Go2. Bring your own MJCF and
            train it here, or drop it straight into <OnePolicy /> and drive it with your keyboard.
          </p>
        </div>
        <HeroRobot />
      </section>

      <Marquee items={["MuJoCo-Warp", "PPO", "Froude scaling", "4096 robots per batch", ">> and >< knees", "WASD to walk"]} />

      <section className="mx-auto w-full max-w-7xl px-4 pt-12">
        <div className="rounded-sm border-3 border-ink bg-yellow p-6 shadow-brutal sm:p-8">
          <h2 className="font-heading text-3xl uppercase sm:text-4xl">
            What is <OnePolicy />?
          </h2>
          <p className="mt-4 max-w-4xl text-lg font-medium">
            <OnePolicy /> is a single locomotion policy that walks a huge range of quadrupeds: from a one-ton
            giant down to a robot weighing a few grams, with leg lengths that differ by more than 25×. There is no
            per-robot training and no fine-tuning. The same network reads each robot&apos;s body and limits and
            works out how to move it, so you can change a robot&apos;s size, mass or proportions and it will
            still walk.
          </p>
        </div>
      </section>

      <section className="mx-auto grid w-full max-w-7xl gap-8 px-4 py-12 md:grid-cols-2">
        <ChoiceCard
          index={0}
          big
          color="bg-pink"
          icon={Dumbbell}
          title="Train your own robot"
          subtitle="Upload an MJCF, tweak rewards, limits and morphology conditions, then train on your own GPU and watch it learn live."
          tag="runs on this machine"
          onClick={() => router.push("/train/")}
        />
        <ChoiceCard
          index={1}
          big
          color="bg-cyan"
          icon={Bot}
          title={<OnePolicy className="text-6xl" />}
          subtitle="Pick a bundled robot, generate one with sliders, or upload yours. Then drop it in the sim and drive it with WASD."
          tag="no training needed"
          onClick={() => router.push("/onepolicy/")}
        />
      </section>
    </>
  );
}
