export function Marquee({ items, className = "bg-ink text-paper" }: { items: string[]; className?: string }) {
  const row = [...items, ...items];
  return (
    <div className={`overflow-hidden border-y-3 border-ink py-2 ${className}`}>
      <div className="animate-marquee flex w-max gap-8 font-heading text-lg whitespace-nowrap uppercase">
        {row.map((t, i) => (
          <span key={i} className="flex items-center gap-8">
            {t} <span className="text-yellow">✦</span>
          </span>
        ))}
      </div>
    </div>
  );
}
