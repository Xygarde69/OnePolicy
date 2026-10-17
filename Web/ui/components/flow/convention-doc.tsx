import { BookOpen } from "lucide-react";

const RULES = [
  ["One floating base", "A body with a free joint (<freejoint/> or <joint type=\"free\"/>)."],
  ["4 legs × 3 hinges", "Ordered from the body out: abduction (about the long axis), hip and knee (about the sideways axis). No other joints."],
  ["Feet", "A collision geom on each foot: the lowest one below each knee is taken."],
  ["A standing pose", "Zero angles = legs straight down (Go2, ANYmal, Spot), or a standing <keyframe>."],
  ["The robot file", "Not a scene.xml that <include>s it (the fixer swaps it in for you)."],
];

/** The adapter's convention in five lines (README.md "Robot convention"). */
export function ConventionDoc() {
  return (
    <aside className="brutal flex flex-col gap-4 bg-yellow p-5">
      <div className="flex items-center gap-2">
        <BookOpen className="size-5" strokeWidth={2.5} />
        <span className="font-heading text-xl uppercase">The convention</span>
      </div>
      <ol className="flex flex-col gap-3">
        {RULES.map(([title, text], i) => (
          <li key={title} className="flex gap-3">
            <span className="flex size-7 shrink-0 items-center justify-center rounded-sm border-2 border-ink bg-white font-mono text-sm font-bold">
              {i + 1}
            </span>
            <div className="text-sm">
              <div className="font-bold uppercase">{title}</div>
              <div>{text}</div>
            </div>
          </li>
        ))}
      </ol>
      <p className="rounded-sm border-2 border-ink bg-white p-3 text-xs">
        Names, leg order, joint signs and which axis faces forward don&apos;t matter: the adapter renames to Go2&apos;s
        names, reorders legs to FR FL RR RL, flips axes and rotates the base. Every change is written to
        <code className="font-mono"> &lt;robot&gt;.temp.xml</code> with what to change in your file.
      </p>
    </aside>
  );
}
