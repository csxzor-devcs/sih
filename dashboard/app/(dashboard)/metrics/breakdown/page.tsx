import fs from "node:fs/promises";
import path from "node:path";

const ATTACK_CLASSES = [
  "DoS Hulk", "PortScan", "DDoS", "DoS GoldenEye",
  "FTP-Patator", "SSH-Patator", "DoS slowloris", "Web Attack", "BENIGN",
];

async function loadBreakdown() {
  const p = path.join(process.cwd(), "..", "artifacts", "eval", "seed_0.json");
  try {
    const raw = await fs.readFile(p, "utf-8");
    return JSON.parse(raw);
  } catch { return null; }
}

export default async function BreakdownPage() {
  const data = await loadBreakdown();
  return (
    <main className="p-6 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-4">Per-Class Breakdown</h1>
      <h2 className="text-2xl font-semibold mb-2">Per-Class AUROC</h2>
      {data ? (
        <table className="w-full border-collapse">
          <thead>
            <tr className="border-b">
              <th className="p-2 text-left">Class</th>
              <th className="p-2 text-left">AUROC</th>
              <th className="p-2 text-left">AUPRC</th>
            </tr>
          </thead>
          <tbody>
            {ATTACK_CLASSES.map((c) => (
              <tr key={c} className="border-b">
                <td className="p-2 font-mono text-sm">{c}</td>
                <td className="p-2 font-mono text-sm">{(data.primary?.[`auroc_${c}`] ?? 0).toFixed(4)}</td>
                <td className="p-2 font-mono text-sm">{(data.primary?.[`auprc_${c}`] ?? 0).toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="text-amber-600">No metrics found.</p>
      )}
    </main>
  );
}
