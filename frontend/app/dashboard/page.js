'use client';

import { useState, useEffect } from 'react';

export default function Dashboard() {
  const [signals, setSignals] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('https://moonwire-backend-production.up.railway.app/api/signal/history?limit=10')
      .then(res => res.json())
      .then(data => setSignals(data.signals || []))
      .catch(err => console.error('Error:', err));

    fetch('https://moonwire-backend-production.up.railway.app/api/signal/stats')
      .then(res => res.json())
      .then(data => {
        setStats(data);
        setLoading(false);
      })
      .catch(err => {
        console.error('Error:', err);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <div className="text-white text-xl">Loading...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-900">
      <div className="container mx-auto px-4 py-8">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-white">Dashboard</h1>
          <div className="flex gap-4">
            
          </div>
        </div>

        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div className="bg-slate-800 rounded-lg p-6">
              <div className="text-slate-400 text-sm mb-2">Total Signals</div>
              <div className="text-3xl font-bold text-white">{stats.total_signals}</div>
            </div>
            <div className="bg-slate-800 rounded-lg p-6">
              <div className="text-slate-400 text-sm mb-2">Win Rate</div>
              <div className="text-3xl font-bold text-green-400">
                {stats.win_rate ? (stats.win_rate * 100).toFixed(1) : 0}%
              </div>
            </div>
            <div className="bg-slate-800 rounded-lg p-6">
              <div className="text-slate-400 text-sm mb-2">Avg Return</div>
              <div className="text-3xl font-bold text-blue-400">
                {stats.avg_outcome ? (stats.avg_outcome * 100).toFixed(2) : 0}%
              </div>
            </div>
            <div className="bg-slate-800 rounded-lg p-6">
              <div className="text-slate-400 text-sm mb-2">Pending</div>
              <div className="text-3xl font-bold text-slate-300">{stats.pending}</div>
            </div>
          </div>
        )}

        <div className="bg-slate-800 rounded-lg p-6">
          <h2 className="text-2xl font-bold text-white mb-6">Recent Signals</h2>
          {signals.length === 0 ? (
            <div className="text-center text-slate-400 py-8">
              No signals yet. Start your backend to see signals!
            </div>
          ) : (
            <div className="space-y-4">
              {signals.map((signal) => (
                <div key={signal.id} className="bg-slate-700 rounded-lg p-4">
                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-4">
                      <div className="text-2xl">
                        {signal.direction === 'long' ? '🟢' : '🔴'}
                      </div>
                      <div>
                        <div className="text-white font-semibold text-lg">{signal.symbol}</div>
                        <div className="text-slate-400 text-sm">{signal.ts}</div>
                      </div>
                    </div>
                    <div className="flex gap-8">
                      <div>
                        <div className="text-slate-400 text-xs">Confidence</div>
                        <div className="text-white font-semibold">
                          {(signal.confidence * 100).toFixed(0)}%
                        </div>
                      </div>
                      <div>
                        <div className="text-slate-400 text-xs">Price</div>
                        <div className="text-white font-semibold">${signal.price.toFixed(2)}</div>
                      </div>
                      <div>
                        <div className="text-slate-400 text-xs">Outcome</div>
                        <div className="text-slate-400">
                          {signal.outcome === null ? 'Pending' : `${(signal.outcome * 100).toFixed(2)}%`}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
