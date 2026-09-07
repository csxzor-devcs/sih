"use client";
import { useState } from "react";

export default function SimulatorPage() {
  const [bytes, setBytes] = useState(1000);
  const [pkts, setPkts] = useState(20);
  const [duration, setDuration] = useState(60);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  async function runForecast() {
    setLoading(true);
    try {
      const r = await fetch("/predict", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          window: {
            // Build a synthetic 12-bin window with the chosen stats
            bins: Array.from({ length: 12 }, () => ({
              bytes_in: bytes,
              bytes_out: bytes * 0.5,
              pkts_in: pkts,
              pkts_out: pkts * 0.5,
              duration,
            })),
          },
          horizon_minutes: 5,
        }),
      });
      setResult(await r.json());
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="p-6 max-w-3xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">What-If Simulator</h1>
      <p className="text-gray-600 mb-6">
        Adjust the synthetic flow stats and see how the model forecasts attack onset.
      </p>

      <div className="space-y-4 mb-6">
        <label className="block">
          <span className="text-sm font-medium">Bytes per bin</span>
          <input type="range" min={100} max={100000} value={bytes}
                 onChange={(e) => setBytes(+e.target.value)} className="w-full" />
          <span className="text-xs text-gray-500">{bytes}</span>
        </label>
        <label className="block">
          <span className="text-sm font-medium">Packets per bin</span>
          <input type="range" min={1} max={500} value={pkts}
                 onChange={(e) => setPkts(+e.target.value)} className="w-full" />
          <span className="text-xs text-gray-500">{pkts}</span>
        </label>
        <label className="block">
          <span className="text-sm font-medium">Duration (s)</span>
          <input type="range" min={1} max={300} value={duration}
                 onChange={(e) => setDuration(+e.target.value)} className="w-full" />
          <span className="text-xs text-gray-500">{duration}</span>
        </label>
      </div>

      <button onClick={runForecast} disabled={loading}
              className="px-4 py-2 bg-blue-600 text-white rounded">
        {loading ? "Forecasting..." : "Forecast"}
      </button>

      {result && (
        <pre className="mt-6 bg-gray-50 p-4 rounded text-sm overflow-auto">
{JSON.stringify(result, null, 2)}
        </pre>
      )}
    </main>
  );
}
