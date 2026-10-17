import { MorphologyChooser } from "@/components/brutal/morphology-chooser";

export default function OnePolicyPage() {
  return (
    <MorphologyChooser
      kicker="2 · onepolicy ;)"
      title="What should OnePolicy walk?"
      intro="OnePolicy/OnePolicy.pt was never trained on your robot: it adapts to the body it's given."
      next="/onepolicy/quadruped/"
      color="bg-cyan"
    />
  );
}
