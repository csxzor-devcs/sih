import fs from "node:fs/promises";
import path from "node:path";

async function loadMetrics() {
  const p = path.join(process.cwd(), "..", "artifacts", "eval", "seed_0.json");
  try {
    const raw = await fs.readFile(p, "utf-8");
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

export default async function ModelCardPage() {
  const metrics = await loadMetrics();
  return (
    <main className="p-6 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-4">Model Card</h1>

      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-2">Architecture</h2>
        <p>
          GRU-based latent dynamics model with a deterministic MLP transition
          z<sub>t+1</sub> = f<sub>θ</sub>(z<sub>t</sub>). Three heads predict
          onset, class, and present. See spec §15.
        </p>
      </section>

      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-2">Training Data</h2>
        <p>CIC-IDS-2017. Temporal split: train days 1-3, val day 4, test day 5 (held-out).</p>
      </section>

      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-2">Metrics</h2>
        {metrics ? (
          <table className="w-full border-collapse">
            <tbody>
              {Object.entries(metrics.primary ?? {}).map(([k, v]) => (
                <tr key={k} className="border-b">
                  <td className="p-2 font-mono text-sm">{k}</td>
                  <td className="p-2 font-mono text-sm">{(v as number).toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="text-amber-600">No metrics file found. Run scripts/eval.py first.</p>
        )}
      </section>

      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-2">Limitations</h2>
        <ul className="list-disc pl-6">
          <li>Trained on CIC-IDS-2017 only; generalization to other datasets is an open question (V3 cross-dataset is Tier 3).</li>
          <li>Benign class is implicitly the absence of any attack class. Heavy class imbalance may bias the present head toward BENIGN.</li>
          <li>Latent dynamics is deterministic; uncertainty is not modeled. This is a known Tier 3 extension (DKF).</li>
        </ul>
      </section>

      <section>
        <h2 className="text-2xl font-semibold mb-2">Intended Use</h2>
        <p>
          Research and demonstration. Not for production deployment without further
          validation on the target network.
        </p>
      </section>
    </main>
  );
}
