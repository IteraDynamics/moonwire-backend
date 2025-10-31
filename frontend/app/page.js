export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
      <div className="container mx-auto px-4 py-20">
        <h1 className="text-6xl font-bold text-center mb-6">MoonWire</h1>
        <p className="text-xl text-center text-slate-300 mb-12">
          Premium crypto signals launching soon
        </p>
        
        <div className="max-w-4xl mx-auto text-center">
          <div className="grid md:grid-cols-3 gap-6 mb-12">
            <div className="bg-slate-800 p-6 rounded-lg">
              <div className="text-3xl font-bold text-emerald-400">54-60%</div>
              <div className="text-slate-400">Win Rate</div>
            </div>
            <div className="bg-slate-800 p-6 rounded-lg">
              <div className="text-3xl font-bold text-blue-400">1.41-1.44</div>
              <div className="text-slate-400">Profit Factor</div>
            </div>
            <div className="bg-slate-800 p-6 rounded-lg">
              <div className="text-3xl font-bold text-purple-400">7-Fold</div>
              <div className="text-slate-400">Validated</div>
            </div>
          </div>
          
          <h2 className="text-3xl font-bold mb-8">Coming Soon</h2>
          <p className="text-slate-300 text-lg">
            Join the waitlist for early access
          </p>
        </div>
      </div>
    </div>
  )
}
