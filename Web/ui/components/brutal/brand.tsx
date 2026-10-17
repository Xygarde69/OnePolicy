import { cn } from "@/lib/utils";

/** The OnePolicy™ wordmark: curvy script, ™ raised. Use it wherever the name is shown. */
export function OnePolicy({ className }: { className?: string }) {
  return (
    <span className={cn("font-script font-normal tracking-normal normal-case whitespace-nowrap", className)}>
      OnePolicy
      <sup className="ml-[0.05em] align-super font-sans text-[0.4em] font-bold">™</sup>
    </span>
  );
}
