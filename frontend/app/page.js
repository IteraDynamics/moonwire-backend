'use client'

import { useState } from 'react'
import { CheckIcon } from '@heroicons/react/24/solid'

export default function Home() {
  const [billingPeriod, setBillingPeriod] = useState<'monthly' | 'annual'>('monthly')

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Hero Section */}
      <header className="container mx-auto px-4 py-20 text-center">
        <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 tracking-tight">
          MoonWire
        </h1>
        <p className="text-xl md:text-2xl text-slate-300 mb-4 max-w-3xl mx-auto">
          Premium crypto signals with institutional-grade validation
        </p>
        <p className="text-lg text-slate-400 mb-12 max-w-2xl mx-auto">
          54-60% win rate validated across 7 time periods. Not for everyone. For traders who want an edge.
        </p>
        
        {/* Stats Bar */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 max-w-4xl mx-auto mb-16">
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-6 border border-slate-700">
            <div className="text-3xl font-bold text-emerald-400 mb-2">54-60%</div>
            <div className="text-sm text-slate-400">Win Rate</div>
          </div>
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-6 border border-slate-700">
            <div className="text-3xl font-bold text-blue-400 mb-2">1.41-1.44</div>
            <div className="text-sm text-slate-400">Profit Factor</div>
          </div>
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-6 border border-slate-700">
            <div className="text-3xl font-bold text-purple-400 mb-2">-13%</div>
            <div className="text-sm text-slate-400">Max Drawdown</div>
          </div>
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-6 border border-slate-700">
            <div className="text-3xl font-bold text-amber-400 mb-2">7-Fold</div>
            <div className="text-sm text-slate-400">Validated</div>
          </div>
        </div>
      </header>

      {/* Pricing Section */}
      <section className="container mx-auto px-4 py-16">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Choose Your Edge
          </h2>
          <p className="text-slate-400 text-lg">
            Two tiers. One goal: Profitable trading.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          {/* Standard Tier */}
          <div className="bg-slate-800/70 backdrop-blur-sm rounded-2xl p-8 border border-slate-700 hover:border-emerald-500 transition-all hover:scale-105">
            <div className="mb-6">
              <h3 className="text-2xl font-bold text-white mb-2">Standard</h3>
              <p className="text-slate-400">Balanced activity & quality</p>
            </div>
            
            <div className="mb-6">
              <div className="flex items-baseline">
                <span className="text-5xl font-bold text-white">$79</span>
                <span className="text-slate-400 ml-2">/month</span>
              </div>
            </div>

            <div className="space-y-4 mb-8">
              <div className="flex items-start">
                <CheckIcon className="w-5 h-5 text-emerald-400 mr-3 mt-0.5 flex-shrink-0" />
                <span className="text-slate-300">54.7% win rate</span>
              </div>
              <div className="flex items-start">
                <CheckIcon className="w-5 h-5 text-emerald-400 mr-3 mt-0.5 flex-shrink-0" />
                <span className="text-slate-300">1.41 profit factor</span>
              </div>
              <div className="flex items-start">
                <CheckIcon className="w-5 h-5 text-emerald-400 mr-3 mt-0.5 flex-shrink-0" />
                <span className="text-slate-300">~15 signals per month</span>
              </div>
              <div className="flex items-start">
                <CheckIcon className="w-5 h-5 text-emerald-400 mr-3 mt-0.5 flex-shrink-0" />
                <span className="text-slate-300">-13% max drawdown (best risk control)</span>
              </div>
              <div className="flex items-start">
                <CheckIcon className="w-5 h-5 text-emerald-400 mr-3 mt-0.5 flex-shrink-0" />
                <span className="text-slate-300">BTC + ETH coverage</span>
              </div>
              <div className="flex items-start">
                <CheckIcon className="w-5 h-5 text-emerald-400 mr-3 mt-0.5 flex-shrink-0" />
                <span className="text-slate-300">Discord + Email alerts</span>
              </div>
            </div>

            
              href="https://buy.stripe.com/YOUR_STANDARD_LINK"
              className="block w-full bg-emerald-600 hover:bg-emerald-500 text-white font-semibold py-3 px-6 rounded-lg text-center transition-colors"
            >
              Get Started
            </a>
          </div>

          {/* Elite Tier */}
          <div className="bg-gradient-to-br from-blue-600/20 to-purple-600/20 backdrop-blur-sm rounded-2xl p-8 border-2 border-blue-500 relative hover:scale-105 transition-all">
            <div className="absolute -top-4 left-1/2 transform -translate-x-1/2 bg-blue-600 text-white px-4 py-1 rounded-full text-sm font-semibold">
              MOST SELECTIVE
            </div>
            
            <div className="mb-6">
              <h3 className="text-2xl font-bold text-white mb-2">Elite</h3>
              <p className="text-slate-300">Ultra-selective, highest win rate</p>
            </div>
            
            <div className="mb-6">
              <div className="flex items-baseline">
                <span className="text-5xl font-bold text-white">$129</span>
                <span className="text-slate-300 ml-2">/month</span>
              </div>
            </div>

            <div className="space-y-4 mb-8">
              <div className="flex items-start">
                <CheckIcon className="w-5 h-5 text-blue-400 mr-3 mt-0.5 flex-shrink-0" />
                <span className="text-slate-200">59.5% win rate</span>
              </div>
              <div className="flex items-start">
                <CheckIcon className="w-5 h-5 text-blue-400 mr-3 mt-0.5 flex-shrink-0" />
                <span className="text-slate-200">1.44 profit factor</span>
              </div>
              <div className="flex items-start">
                <CheckIcon className="w-5 h-5 text-blue-400 mr-3 mt-0.5 flex-shrink-0" />
                <span className="text-slate-200">~11 signals per month</span>
              </div>
              <div className="flex items-start">
                <CheckIcon className="w-5 h-5 text-blue-400 mr-3 mt-0.5 flex-shrink-0" />
                <span className="text-slate-200">BTC: 75% win rate</span>
              </div>
              <div className="flex items-start">
                <CheckIcon className="w-5 h-5 text-blue-400 mr-3 mt-0.5 flex-shrink-0" />
                <span className="text-slate-200">Only highest conviction setups</span>
              </div>
              <div className="flex items-start">
                <CheckIcon className="w-5 h-5 text-blue-400 mr-3 mt-0.5 flex-shrink-0" />
                <span className="text-slate-200">Priority support</span>
              </div>
            </div>

            
              href="https://buy.stripe.com/YOUR_ELITE_LINK"
              className="block w-full bg-blue-600 hover:bg-blue-500 text-white font-semibold py-3 px-6 rounded-lg text-center transition-colors"
            >
              Go Elite
            </a>
          </div>

          {/* Bundle Tier */}
          <div className="bg-slate-800/70 backdrop-blur-sm rounded-2xl p-8 border border-slate-700 hover:border-purple-500 transition-all hover:scale-105">
            <div className="mb-6">
              <h3 className="text-2xl font-bold text-white mb-2">Bundle</h3>
              <p className="text-slate-400">Maximum coverage</p>
            </div>
            
            <div className="mb-6">
              <div className="flex items-baseline">
                <span className="text-5xl font-bold text-white">$179</span>
                <span className="text-slate-400 ml-2">/month</span>
              </div>
              <div className="text-emerald-400 text-sm font-semibold mt-2">
                Save $29/month
              </div>
            </div>

            <div className="space-y-4 mb-8">
              <div className="flex items-start">
                <CheckIcon className="w-5 h-5 text-purple-400 mr-3 mt-0.5 flex-shrink-0" />
                <span className="text-slate-300">All Standard features</span>
              </div>
              <div className="flex items-start">
                <CheckIcon className="w-5 h-5 text-purple-400 mr-3 mt-0.5 flex-shrink-0" />
                <span className="text-slate-300">All Elite features</span>
              </div>
              <div className="flex items-start">
                <CheckIcon className="w-5 h-5 text-purple-400 mr-3 mt-0.5 flex-shrink-0" />
                <span className="text-slate-300">~26 total signals/month</span>
              </div>
              <div className="flex items-start">
                <CheckIcon className="w-5 h-5 text-purple-400 mr-3 mt-0.5 flex-shrink-0" />
                <span className="text-slate-300">Both signal tiers</span>
              </div>
              <div className="flex items-start">
                <CheckIcon className="w-5 h-5 text-purple-400 mr-3 mt-0.5 flex-shrink-0" />
                <span className="text-slate-300">Maximum edge</span>
              </div>
              <div className="flex items-start">
                <CheckIcon className="w-5 h-5 text-purple-400 mr-3 mt-0.5 flex-shrink-0" />
                <span className="text-slate-300">Best value</span>
              </div>
            </div>

            
              href="https://buy.stripe.com/YOUR_BUNDLE_LINK"
              className="block w-full bg-purple-600 hover:bg-purple-500 text-white font-semibold py-3 px-6 rounded-lg text-center transition-colors"
            >
              Get Bundle
            </a>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="container mx-auto px-4 py-16">
        <h2 className="text-3xl md:text-4xl font-bold text-white text-center mb-12">
          How It Works
        </h2>
        
        <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
          <div className="text-center">
            <div className="bg-emerald-600 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 text-2xl font-bold text-white">
              1
            </div>
            <h3 className="text-xl font-semibold text-white mb-3">Subscribe</h3>
            <p className="text-slate-400">
              Choose your tier. Get instant Discord access. No setup required.
            </p>
          </div>
          
          <div className="text-center">
            <div className="bg-blue-600 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 text-2xl font-bold text-white">
              2
            </div>
            <h3 className="text-xl font-semibold text-white mb-3">Receive Signals</h3>
            <p className="text-slate-400">
              Get real-time alerts via Discord + Email. Entry, exit, confidence included.
            </p>
          </div>
          
          <div className="text-center">
            <div className="bg-purple-600 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 text-2xl font-bold text-white">
              3
            </div>
            <h3 className="text-xl font-semibold text-white mb-3">Trade & Profit</h3>
            <p className="text-slate-400">
              Execute signals on your exchange. Track performance. Adjust risk as needed.
            </p>
          </div>
        </div>
      </section>

      {/* Validation Section */}
      <section className="container mx-auto px-4 py-16">
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl p-12 border border-slate-700 max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold text-white text-center mb-6">
            Walk-Forward Validated
          </h2>
          <p className="text-slate-300 text-center mb-8 text-lg">
            We don't cherry-pick results. Every stat you see is validated across 7 different time periods using proper walk-forward testing.
          </p>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-slate-900/50 rounded-lg p-6">
              <h3 className="font-semibold text-white mb-3">Standard Tier</h3>
              <ul className="space-y-2 text-slate-300 text-sm">
                <li>• 270 days of training data</li>
                <li>• Regime-filtered (only trending markets)</li>
                <li>• 7-fold time-series validation</li>
                <li>• Consistent 54.7% WR across all folds</li>
              </ul>
            </div>
            
            <div className="bg-slate-900/50 rounded-lg p-6">
              <h3 className="font-semibold text-white mb-3">Elite Tier</h3>
              <ul className="space-y-2 text-slate-300 text-sm">
                <li>• 365 days of training data</li>
                <li>• Ultra-selective regime filtering</li>
                <li>• 7-fold time-series validation</li>
                <li>• Consistent 59.5% WR, 75% on BTC</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* FAQ */}
      <section className="container mx-auto px-4 py-16">
        <h2 className="text-3xl md:text-4xl font-bold text-white text-center mb-12">
          Frequently Asked Questions
        </h2>
        
        <div className="max-w-3xl mx-auto space-y-6">
          <details className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-6 border border-slate-700">
            <summary className="font-semibold text-white cursor-pointer">
              What's the difference between Standard and Elite?
            </summary>
            <p className="mt-4 text-slate-400">
              Standard gives you more frequent signals (15/month) with 54.7% win rate. Elite is ultra-selective (11/month) with 59.5% win rate. Both are profitable - Standard is for active traders, Elite is for patient capital.
            </p>
          </details>
          
          <details className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-6 border border-slate-700">
            <summary className="font-semibold text-white cursor-pointer">
              How do I receive signals?
            </summary>
            <p className="mt-4 text-slate-400">
              Real-time alerts via Discord (instant notifications) and email backup. Each signal includes: Asset, Direction (long/short), Entry price, Confidence score, and suggested risk management.
            </p>
          </details>
          
          <details className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-6 border border-slate-700">
            <summary className="font-semibold text-white cursor-pointer">
              What's your refund policy?
            </summary>
            <p className="mt-4 text-slate-400">
              7-day money-back guarantee. If you're not satisfied, email us within 7 days for a full refund. No questions asked.
            </p>
          </details>
          
          <details className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-6 border border-slate-700">
            <summary className="font-semibold text-white cursor-pointer">
              How much capital do I need?
            </summary>
            <p className="mt-4 text-slate-400">
              Minimum recommended: $1,000-$2,000 for proper position sizing (1-2% risk per trade). You can start smaller but risk management becomes harder.
            </p>
          </details>
          
          <details className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-6 border border-slate-700">
            <summary className="font-semibold text-white cursor-pointer">
              Is this financial advice?
            </summary>
            <p className="mt-4 text-slate-400">
              No. MoonWire provides trading signals for informational purposes only. We are not financial advisors. Trade at your own risk. Past performance does not guarantee future results.
            </p>
          </details>
        </div>
      </section>

      {/* CTA Section */}
      <section className="container mx-auto px-4 py-20 text-center">
        <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
          Ready to get an edge?
        </h2>
        <p className="text-xl text-slate-300 mb-8 max-w-2xl mx-auto">
          Join traders using institutional-grade signals. Start with 7-day money-back guarantee.
        </p>
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          
            href="https://buy.stripe.com/YOUR_STANDARD_LINK"
            className="bg-emerald-600 hover:bg-emerald-500 text-white font-semibold py-4 px-8 rounded-lg transition-colors text-lg"
          >
            Start Standard ($79/mo)
          </a>
          
            href="https://buy.stripe.com/YOUR_ELITE_LINK"
            className="bg-blue-600 hover:bg-blue-500 text-white font-semibold py-4 px-8 rounded-lg transition-colors text-lg"
          >
            Go Elite ($129/mo)
          </a>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-slate-800 py-12">
        <div className="container mx-auto px-4 text-center text-slate-500 text-sm">
          <p className="mb-2">© 2025 MoonWire. All rights reserved.</p>
          <p>Trading involves risk. Past performance does not guarantee future results.</p>
        </div>
      </footer>
    </div>
  )
}