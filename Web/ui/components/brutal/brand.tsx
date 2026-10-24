import { cn } from "@/lib/utils";

/** The OnePolicy™ wordmark: curvy script, ™ raised. Use it wherever the name is shown.
 *  `r`: OnePolicy_r, the rough-terrain policy. */
export function OnePolicy({ className, r }: { className?: string; r?: boolean }) {
  return (
    <span className={cn("font-script font-normal tracking-normal normal-case whitespace-nowrap", className)}>
      OnePolicy{r && "_r"}
      <sup className="ml-[0.05em] align-super font-sans text-[0.4em] font-bold">™</sup>
    </span>
  );
}
