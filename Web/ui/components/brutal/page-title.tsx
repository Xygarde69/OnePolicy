import { cn } from "@/lib/utils";

export function PageTitle({ kicker, title, children, color = "bg-yellow" }: {
  kicker: string;
  title: string;
  children?: React.ReactNode;
  color?: string;
}) {
  return (
    <div className="flex flex-col gap-3">
      <span className={cn("w-fit rounded-sm border-2 border-ink px-2 py-0.5 font-mono text-xs font-bold uppercase", color)}>
        {kicker}
      </span>
      <h1 className="font-heading text-4xl leading-[0.95] uppercase sm:text-6xl">{title}</h1>
      {children && <p className="max-w-2xl text-lg font-medium">{children}</p>}
    </div>
  );
}
