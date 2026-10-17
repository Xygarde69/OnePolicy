import { OnePolicy } from "@/components/brutal/brand";
import { MorphologyChooser } from "@/components/brutal/morphology-chooser";

export default function OnePolicyPage() {
  return (
    <MorphologyChooser
      kicker={<>2 · <OnePolicy className="text-sm" /></>}
      title={<>What should <OnePolicy /> walk?</>}
      intro={<><OnePolicy /> was never trained on your robot: it adapts to the body it&apos;s given.</>}
      next="/onepolicy/quadruped/"
      color="bg-cyan"
    />
  );
}
