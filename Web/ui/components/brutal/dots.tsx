/** "..." where each dot blinks in turn. */
export function Dots({ className }: { className?: string }) {
  return (
    <span className={className} aria-hidden>
      {[0, 1, 2].map((i) => (
        <span key={i} className="animate-dot" style={{ animationDelay: `${i * 0.4}s` }}>.</span>
      ))}
    </span>
  );
}
