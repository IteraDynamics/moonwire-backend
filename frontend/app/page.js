'use client'

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <header className="container mx-auto px-4 py-20 text-center">
        <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 tracking-tight">
          MoonWire
        </h1>
        <p className="text-xl md:text-2xl text-slate-300 mb-4 max-w-3xl mx-auto">
          Premium crypto signals with institutional-grade validation
        </p>
        <p className="text-lg text-slate-400 mb-12 max-w-2xl mx-auto">
          54-60% win rate validated across 7 time periods
        </p>
        
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

      <section className="container mx-auto px-4 py-16">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Choose Your Edge</h2>
          <p className="text-slate-400 text-lg">Two tiers. One goal: Profitable trading.</p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          <PricingCard 
            title="Standard"
            subtitle="Balanced activity & quality"
            price="79"
            features={[
              "54.7% win rate",
              "1.41 profit factor",
              "~15 signals per month",
              "-13% max drawdown",
              "BTC + ETH coverage",
              "Discord + Email alerts"
            ]}
            href="https://buy.stripe.com/test_28EdR96aM95fdktcD40gw02"
            buttonText="Get Started"
            color="emerald"
          />

          <PricingCard 
            title="Elite"
            subtitle="Ultra-selective, highest win rate"
            price="129"
            badge="MOST SELECTIVE"
            features={[
              "59.5% win rate",
              "1.44 profit factor",
              "~11 signals per month",
              "BTC: 75% win rate",
              "Only highest conviction",
              "Priority support"
            ]}
            href="https://buy.stripe.com/test_7sYbJ1fLmgxHcgp6eG0gw01"
            buttonText="Go Elite"
            color="blue"
            featured={true}
          />

          <PricingCard 
            title="Bundle"
            subtitle="Maximum coverage"
            price="179"
            savings="Save $29/month"
            features={[
              "All Standard features",
              "All Elite features",
              "~26 total signals/month",
              "Both signal tiers",
              "Maximum edge",
              "Best value"
            ]}
            href="https://buy.stripe.com/test_7sYbJ1eHi3KV5S1dH80gw00"
            buttonText="Get Bundle"
            color="purple"
          />
        </div>
      </section>

      <section className="container mx-auto px-4 py-16">
        <h2 className="text-3xl md:text-4xl font-bold text-white text-center mb-12">How It Works</h2>
        <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
          <Step number="1" title="Subscribe" description="Choose your tier. Get instant Discord access. No setup required." color="emerald" />
          <Step number="2" title="Receive Signals" description="Get real-time alerts via Discord. Entry, exit, confidence included." color="blue" />
          <Step number="3" title="Trade & Profit" description="Execute signals on your exchange. Track performance." color="purple" />
        </div>
      </section>

      <footer className="border-t border-slate-800 py-12">
        <div className="container mx-auto px-4 text-center text-slate-500 text-sm">
          <p className="mb-2">© 2025 MoonWire. All rights reserved.</p>
          <p>Trading involves risk. Past performance does not guarantee future results.</p>
        </div>
      </footer>
    </div>
  )
}

function PricingCard({ title, subtitle, price, savings, badge, features, href, buttonText, color, featured }) {
  const colorClasses = {
    emerald: 'bg-emerald-600 hover:bg-emerald-500 border-emerald-500 text-emerald-400',
    blue: 'bg-blue-600 hover:bg-blue-500 border-blue-500 text-blue-400',
    purple: 'bg-purple-600 hover:bg-purple-500 border-purple-500 text-purple-400'
  }
  
  const borderColor = featured ? `border-2 border-${color}-500` : 'border border-slate-700'
  const hoverBorder = featured ? '' : `hover:border-${color}-500`
  
  return (
    <div className={`bg-slate-800/70 backdrop-blur-sm rounded-2xl p-8 ${borderColor} ${hoverBorder} transition-all hover:scale-105 relative`}>
      {badge && (
        <div className={`absolute -top-4 left-1/2 transform -translate-x-1/2 bg-${color}-600 text-white px-4 py-1 rounded-full text-sm font-semibold`}>
          {badge}
        </div>
      )}
      <div className="mb-6">
        <h3 className="text-2xl font-bold text-white mb-2">{title}</h3>
        <p className={featured ? 'text-slate-300' : 'text-slate-400'}>{subtitle}</p>
      </div>
      <div className="mb-6">
        <div className="flex items-baseline">
          <span className="text-5xl font-bold text-white">${price}</span>
          <span className={featured ? 'text-slate-300 ml-2' : 'text-slate-400 ml-2'}>/month</span>
        </div>
        {savings && <div className="text-emerald-400 text-sm font-semibold mt-2">{savings}</div>}
      </div>
      <div className="space-y-4 mb-8">
        {features.map((feature, i) => (
          <div key={i} className="flex items-start">
            <span className={`${colorClasses[color].split(' ')[3]} mr-3`}>✓</span>
            <span className={featured ? 'text-slate-200' : 'text-slate-300'}>{feature}</span>
          </div>
        ))}
      </div>
      <a href={href} className={`block w-full ${colorClasses[color].split(' ')[0]} ${colorClasses[color].split(' ')[1]} text-white font-semibold py-3 px-6 rounded-lg text-center transition-colors`}>
        {buttonText}
      </a>
    </div>
  )
}

function Step({ number, title, description, color }) {
  return (
    <div className="text-center">
      <div className={`bg-${color}-600 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 text-2xl font-bold text-white`}>
        {number}
      </div>
      <h3 className="text-xl font-semibold text-white mb-3">{title}</h3>
      <p className="text-slate-400">{description}</p>
    </div>
  )
}
