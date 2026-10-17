import { MorphologyChooser } from "@/components/brutal/morphology-chooser";

export default function TrainPage() {
  return (
    <MorphologyChooser
      kicker="1 · train your own robot"
      title="What are we training?"
      intro="Training runs train.py on this machine's GPU. Your robot trains next to generated quadrupeds, so the policy stays general."
      next="/train/quadruped/"
      color="bg-pink"
    />
  );
}
