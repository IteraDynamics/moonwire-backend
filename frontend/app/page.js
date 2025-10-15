export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800">
      {/* Hero Section */}
      <div className="container mx-auto px-4 py-20">
        <div className="text-center max-w-4xl mx-auto">
          <h1 className="text-5xl md:text-6xl font-bold text-white mb-6">
            Crypto Signals That Actually Work
          </h1>
          <p className="text-xl text-slate-300 mb-8">
            ML-powered trading signals for Bitcoin, Ethereum, and altcoins. 
            Real-time detection. Transparent performance tracking.
          </p>
          
          {/* CTA Buttons */}
          <div className="flex gap-4 justify-center">
            <a 
              href="#pricing" 
              className="px-8 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold transition"
            >
              Get Started
            </a>
            <a 
              href="https://discord.gg/DMVJYPj5" 
              className="px-8 py-3 bg-slate-700 hover:bg-slate-600 text-white rounded-lg font-semibold transition"
            >
              Join Discord
            </a>
          </div>
        </div>

        {/* Stats Section */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-20 text-center">
          <div className="bg-slate-800 rounded-lg p-6">
            <div className="text-4xl font-bold text-blue-400 mb-2">58%</div>
            <div className="text-slate-300">Win Rate</div>
          </div>
          <div className="bg-slate-800 rounded-lg p-6">
            <div className="text-4xl font-bold text-blue-400 mb-2">+1.2%</div>
            <div className="text-slate-300">Avg Return</div>
          </div>
          <div className="bg-slate-800 rounded-lg p-6">
            <div className="text-4xl font-bold text-blue-400 mb-2">128</div>
            <div className="text-slate-300">Signals Tracked</div>
          </div>
        </div>

        {/* Features Section */}
        <div className="mt-20">
          <h2 className="text-3xl font-bold text-white text-center mb-12">
            How It Works
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="text-4xl mb-4">🤖</div>
              <h3 className="text-xl font-semibold text-white mb-2">ML Detection</h3>
              <p className="text-slate-300">
                Advanced algorithms analyze market data and social sentiment in real-time
              </p>
            </div>
            <div className="text-center">
              <div className="text-4xl mb-4">⚡</div>
              <h3 className="text-xl font-semibold text-white mb-2">Instant Alerts</h3>
              <p className="text-slate-300">
                Get signals in Discord the moment opportunities are detected
              </p>
            </div>
            <div className="text-center">
              <div className="text-4xl mb-4">📊</div>
              <h3 className="text-xl font-semibold text-white mb-2">Track Performance</h3>
              <p className="text-slate-300">
                See transparent win rates and outcomes for every signal
              </p>
            </div>
          </div>
        </div>

        {/* Pricing Section */}
        <div id="pricing" className="mt-20">
          <h2 className="text-3xl font-bold text-white text-center mb-12">
            Simple Pricing
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            {/* Free Tier */}
            <div className="bg-slate-800 rounded-lg p-8 border border-slate-700">
              <h3 className="text-2xl font-bold text-white mb-2">Free</h3>
              <div className="text-4xl font-bold text-white mb-6">$0</div>
              <ul className="space-y-3 mb-8 text-slate-300">
                <li>✓ Daily signals</li>
                <li>✓ 15-minute delay</li>
                <li>✓ Performance history</li>
                <li>✓ Discord access</li>
              </ul>
              <a 
                href="https://discord.gg/DMVJYPj5"
                className="block w-full text-center px-6 py-3 bg-slate-700 hover:bg-slate-600 text-white rounded-lg font-semibold transition"
              >
                Join Free
              </a>
            </div>

            {/* Standard Tier */}
            <div className="bg-blue-600 rounded-lg p-8 border-2 border-blue-400 relative">
              <div className="absolute -top-4 left-1/2 transform -translate-x-1/2 bg-blue-400 text-blue-900 px-4 py-1 rounded-full text-sm font-semibold">
                Popular
              </div>
              <h3 className="text-2xl font-bold text-white mb-2">Standard</h3>
              <div className="text-4xl font-bold text-white mb-6">
                $29<span className="text-lg">/mo</span>
              </div>
              <ul className="space-y-3 mb-8 text-white">
                <li>✓ Real-time signals</li>
                <li>✓ No delay</li>
                <li>✓ Premium Discord</li>
                <li>✓ Signal history</li>
              </ul>
              <a 
                href="/checkout?tier=standard"
                className="block w-full text-center px-6 py-3 bg-white hover:bg-slate-100 text-blue-600 rounded-lg font-semibold transition"
              >
                Get Started
              </a>
            </div>

            {/* Premium Tier */}
            <div className="bg-slate-800 rounded-lg p-8 border border-slate-700">
              <h3 className="text-2xl font-bold text-white mb-2">Premium</h3>
              <div className="text-4xl font-bold text-white mb-6">
                $79<span className="text-lg">/mo</span>
              </div>
              <ul className="space-y-3 mb-8 text-slate-300">
                <li>✓ Everything in Standard</li>
                <li>✓ Custom alerts</li>
                <li>✓ Historical backtests</li>
                <li>✓ Priority support</li>
              </ul>
              <a 
                href="/checkout?tier=premium"
                className="block w-full text-center px-6 py-3 bg-slate-700 hover:bg-slate-600 text-white rounded-lg font-semibold transition"
              >
                Go Premium
              </a>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}