import React, { useState, useEffect } from 'react';
import './WebPage.css';
import heroVideo from '../../assets/videos/hero_video.mp4';
import turtleImg from '../../assets/images/turtle.png';
import orcaImg from '../../assets/images/orca.jpg';
import dugongImg from '../../assets/images/dugong.jpg';
import Grainient from '../../components/Grainient';
import PillNav from '../../components/PillNav/PillNav';

export default function WebPage({ onBack, onContinue }) {
  const marqueeContent = (
    <>
      <span className="star">✦</span> $29.24B <span className="small-text">ECONOMY LOSS</span>
      <span className="star">✦</span> 300,000+ <span className="small-text">CATECEAN DEATHS</span>
      <span className="star">✦</span> 500K-1M TONS <span className="small-text">GHOST GEAR</span>
    </>
  );

  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    window.scrollTo(0, 0);

    const handleScroll = () => {
      if (window.scrollY > 10) {
        setIsScrolled(true);
      } else {
        setIsScrolled(false);
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div className="webpage-root">
      <div className="wave-top-section">
        <video
          className="full-page-video-bg"
          autoPlay
          loop
          muted
          playsInline
          src={heroVideo}
        />

        <nav className={`wave-nav ${isScrolled ? 'scrolled' : ''}`}>
          <div className="wave-logo" onClick={onBack} style={{ cursor: 'pointer' }}>WAVESENTINEL</div>
          <div className="wave-links">
            <a href="#system">SYSTEM</a>
            <a href="#problem">PROBLEM</a>
            <a href="#detection">DETECTION</a>
            <a href="#research">ARTICLES</a>
          </div>
        </nav>

        <section id="system" className="wave-hero">
          <div className="hero-left">
            <div className="wave-subtitle">UNDERWATER ACOUSTIC INTELLIGENCE</div>
            <h1 className="wave-h1">
              <span className="text-white">SHADOWS</span><br />
              <span className="text-gray">IN THE</span><br />
              <span className="text-white">DEEP.</span>
            </h1>
            <p className="wave-hero-p">
              Sunken trash and hidden seabed hazards lie trapped in deep ocean waters. Sitting unseen beneath the surface, this forgotten waste slowly suffocates innocent sea life while secretly tearing fishing nets and damaging boat hulls, causing severe financial heartbreak and danger for coastal families before anyone ever notices.
            </p>
          </div>
          <div className="hero-right">
            <div className="video-placeholder">
              {/* Kept empty to maintain layout spacing */}
            </div>
          </div>
        </section>

        <section className="wave-stats">
          <div className="stat-box">
            <div className="stat-val">90%</div>
            <div className="stat-label">DEBRIS UNSEEN</div>
          </div>
          <div className="stat-box">
            <div className="stat-val">850+</div>
            <div className="stat-label">SPECIES AFFECTED</div>
          </div>
          <div className="stat-box">
            <div className="stat-val">34%</div>
            <div className="stat-label">CORAL TISSUE LOSS</div>
          </div>
          <div className="stat-box">
            <div className="stat-val">60,000 CR</div>
            <div className="stat-label">INDIA'S BLUE ECONOMY</div>
          </div>
        </section>
      </div>

      <div className="wave-yellow-marquee">
        <div className="marquee-content">
          {marqueeContent}
          {marqueeContent}
          {marqueeContent}
          {marqueeContent}
          {marqueeContent}
          {marqueeContent}
          {marqueeContent}
          {marqueeContent}
          {marqueeContent}
          {marqueeContent}
          {marqueeContent}
          {marqueeContent}
        </div>
      </div>

      <section id="problem" className="wave-problem-section">
        <div className="grainient-bg" style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, zIndex: 0 }}>
          <Grainient
            color1="#00a2e2"
            color2="#001ddb"
            color3="#0f0f0f"
            timeSpeed={2}
            colorBalance={0.12}
            warpStrength={3.45}
            warpFrequency={0}
            warpSpeed={3}
            warpAmplitude={35}
            blendAngle={129}
            blendSoftness={1}
            rotationAmount={780}
            noiseScale={1.55}
            grainAmount={0}
            grainScale={0.2}
            grainAnimated
            contrast={1.5}
            gamma={2.25}
            saturation={1.15}
            centerX={-0.63}
            centerY={-0.22}
            zoom={0.75}
          />
        </div>
        <div className="problem-content" style={{ position: 'relative', zIndex: 1 }}>
          <div className="wave-subtitle">THE PROBLEM</div>
          <h2 className="wave-h2">
            <span className="text-white">THE HIDDEN</span><br />
            <span className="text-gray">WRECKAGE</span>
          </h2>

          <div className="image-cards-grid">
            <div className="img-card">
              <div className="img-placeholder">
                <img src="/05o4569845.png" alt="Underwater Death Traps" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
              </div>
              <div className="card-content">
                <h3 className="card-title">UNDERWATER DEATH TRAPS</h3>
                <p className="card-p">Sunken debris and lost fishing gear create hidden hazards along India's coastline. With no reliable way to track the damage, these silent traps continue killing marine species for years after being discarded, largely unnoticed and unmeasured.</p>
              </div>
            </div>
            <div className="img-card">
              <div className="img-placeholder">
                <img src="/85568007.png" alt="Storm-Wrecked Ports" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
              </div>
              <div className="card-content">
                <h3 className="card-title">STORM-WRECKED PORTS</h3>
                <p className="card-p">Tropical cyclones leave behind underwater wreckage that clogs shipping lanes alongside land damage. This debris disrupts harbours, delays vessel movement, and demands lengthy clearance before ports can resume normal trade operations.</p>
              </div>
            </div>
            <div className="img-card">
              <div className="img-placeholder">
                <img src="/0056565.png" alt="Reef Debris Damage" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
              </div>
              <div className="card-content">
                <h3 className="card-title">REEF DEBRIS DAMAGE</h3>
                <p className="card-p">Heavy debris settling on coral reefs causes damage that builds up slowly but severely. These effects threaten fragile marine ecosystems and the fishing communities and island economies that depend on healthy reefs to survive.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="detection" className="wave-protection-section">
        <div className="wave-protection-title">
          <div className="wave-subtitle">UNRECORDED - UNMEASURED - UNMONITORED</div>
          <h2 className="wave-h2">
            <span className="text-white">THE DEPTHS </span><br />
            <span className="text-gray">NO ONE MONITORS.</span>
          </h2>
        </div>

        <div className="protection-steps-wrapper">
          <div className="protection-step step-1">
            <div className="detect-container">
              <span className="detect-number">01</span>
              <div className="detect-details">
                <div className="detect-label">DETECTION</div>
                <h3 className="detect-title">INVISIBLE MARINE MURDERS</h3>
                <p className="detect-threat">Because missing nets, ropes, and cables continue to trap marine animals long after fishing vessels discard the equipment, the number of marine animal deaths is increasing. When dolphins and turtles come to the surface to breathe, they become entangled, are unable to breathe, and drown, unnoticed and unreported until entire populations begin to silently vanish.</p>
              </div>
            </div>
          </div>

          <div className="protection-step step-2">
            <div className="detect-container">
              <span className="detect-number">02</span>
              <div className="detect-details">
                <div className="detect-label">CLASSIFICATION</div>
                <h3 className="detect-title">SHIPPING DELAYS</h3>
                <p className="detect-threat">Because cyclones dump containers, shattered concrete, and sunken boats directly into the shipping lanes that ports depend on, harbours are frequently shut down. Each time, ships must wait offshore until personnel verify the bottom is clean, which delays trade and costs ports days' worth of lost revenue.</p>
              </div>
            </div>
          </div>

          <div className="protection-step step-3">
            <div className="detect-container">
              <span className="detect-number">03</span>
              <div className="detect-details">
                <div className="detect-label">OBSERVATION</div>
                <h3 className="detect-title">CORAL COLLAPSE ZONE</h3>
                <p className="detect-threat">Heavy nets carried by currents are tearing up centuries-old coral structures, destroying coral reefs more quickly than they can regenerate. In reef-rich areas like the Gulf of Mannar and the Andamans, where fishing populations rely on the survival of the reef, debris lying on top blocks the sunlight coral needs, destroying it from the base up.</p>
              </div>
            </div>
          </div>

          <div className="protection-step step-4">
            <div className="detect-container">
              <span className="detect-number">04</span>
              <div className="detect-details">
                <div className="detect-label">PROTECTION</div>
                <h3 className="detect-title">THE UNOBSERVED SEAFLOOR</h3>
                <p className="detect-threat">Because this damage occurs deep underwater, far from regular monitoring, where no one is seeing, it keeps getting worse. The damage is only identified after it has already occurred, the animal has drowned, the reef has died, or the port has already wasted time and money. If there is no system in place to identify debris before it buries a reef or obstructs a shipping lane.</p>
              </div>
            </div>
          </div>
        </div>
      </section>
      <section id="research" className="wave-research-section">
        <div className="wave-research-title">
          <div className="wave-subtitle">ARTICLES</div>
          <h2 className="wave-h2">
            <span className="text-white">WHAT'S BEING</span><br />
            <span className="text-gray">REPORTED.</span>
          </h2>
        </div>

        <div className="research-marquee-container">
          <div className="research-marquee-content">
            {[
              {
                quote: `"Researchers spotted 144 animals belonging to 35 species, 13 of them endangered or vulnerable, entangled in abandoned fishing gear."`,
                title: "144 Animals, 35 Species: India's Hidden Ghost Net Toll",
                subtitle: "READ ON MONGABAY INDIA",
                url: "https://india.mongabay.com/2024/04/creatures-entangled-ghost-nets-trap-creatures-small-and-great/"
              },
              {
                quote: `"Sea turtles, fish and marine mammals were the most reported groups entangled in abandoned fishing gear."`,
                title: "98% of India's Entanglements Trace Back to Five Gear Types",
                subtitle: "READ ON SCIENCE DIRECT",
                url: "https://www.sciencedirect.com/science/article/pii/S0006320724000752"
              },
              {
                quote: `"Once lost, fishing gear continues to function, trapping marine life or entangling marine habitat."`,
                title: "Why India's Ghost Gear Fixes Keep Failing",
                subtitle: "READ ON MONGABAY NEWS",
                url: "https://news.mongabay.com/2026/06/indias-fishers-confront-homegrown-ghost-gear-problem/"
              },
              {
                quote: `"Mundra has prohibited vessels from anchoring or drifting around the port until further notice."`,
                title: "One Storm, Every Gujarat Port Shut Down",
                subtitle: "READ ON PORT TECHNOLOGY",
                url: "https://porttechnology.org/news/cyclone-biparjoy-stifles-indias-major-ports"
              },
              {
                quote: `"A severe cyclone Michaung caused siltation in the port area, requiring channel depth restoration."`,
                title: "Buried by a Storm: Krishnapatnam's Channel Dig-Out",
                subtitle: "READ ON MARITIME PROFESSIONAL",
                url: "http://www.maritimeprofessional.com/news/post-cyclone-dredging-underway-india-391883"
              },
              {
                quote: `"The closure of Chennai Ports resulted in a loss of over Rs. 100 crore per day."`,
                title: "₹100 Crore a Day: The Real Cost of a Storm-Shut Port",
                subtitle: "READ ON EASYVESSEL",
                url: "https://easyvessel.com/chennai-port-closure-cyclone-michaung-coastal-impact/"
              },
              {
                quote: `"Marine debris is increasing in the Lakshadweep as well, and it represents exactly how wide the footprint of the commercial fishery is."`,
                title: "1,150 Square Metres of Reef, Buried in Nets",
                subtitle: "READ ON MONGABAY INDIA",
                url: "https://india.mongabay.com/2021/01/ghosts-of-the-gulf-marine-debris-a-threat-to-corals-in-the-gulf-of-mannar/"
              },
              {
                quote: `"Fragmentation caused by these derelict nets impacted species such as Acropora, Pocillopora, and Porites."`,
                title: "Decades-Old Coral, Snapped Apart in Minutes",
                subtitle: "READ ON RESEARCHGATE",
                url: "https://www.researchgate.net/publication/356892712_Impact_of_Ghost_Nets_And_Marine_Debris_on_The_Coral_Reef_Ecosystem_of_Andrott_Island_Lakshadweep_India"
              },
              {
                quote: `"A dedicated team turned ghost-net 'busters' by policing the reefs for lost fishing gear."`,
                title: "Divers Turn Ghost-Net Busters to Save Tamil Nadu's Reefs",
                subtitle: "READ ON UNEP",
                url: "https://www.unep.org/news-and-stories/story/underwater-ghost-busting-save-indian-coral-reefs"
              },
              {
                quote: `"Researchers spotted 144 animals belonging to 35 species, 13 of them endangered or vulnerable, entangled in abandoned fishing gear."`,
                title: "144 Animals, 35 Species: India's Hidden Ghost Net Toll",
                subtitle: "READ ON MONGABAY INDIA",
                url: "https://india.mongabay.com/2024/04/creatures-entangled-ghost-nets-trap-creatures-small-and-great/"
              },
              {
                quote: `"Sea turtles, fish and marine mammals were the most reported groups entangled in abandoned fishing gear."`,
                title: "98% of India's Entanglements Trace Back to Five Gear Types",
                subtitle: "READ ON SCIENCE DIRECT",
                url: "https://www.sciencedirect.com/science/article/pii/S0006320724000752"
              },
              {
                quote: `"Once lost, fishing gear continues to function, trapping marine life or entangling marine habitat."`,
                title: "Why India's Ghost Gear Fixes Keep Failing",
                subtitle: "READ ON MONGABAY NEWS",
                url: "https://news.mongabay.com/2026/06/indias-fishers-confront-homegrown-ghost-gear-problem/"
              },
              {
                quote: `"Mundra has prohibited vessels from anchoring or drifting around the port until further notice."`,
                title: "One Storm, Every Gujarat Port Shut Down",
                subtitle: "READ ON PORT TECHNOLOGY",
                url: "https://porttechnology.org/news/cyclone-biparjoy-stifles-indias-major-ports"
              },
              {
                quote: `"A severe cyclone Michaung caused siltation in the port area, requiring channel depth restoration."`,
                title: "Buried by a Storm: Krishnapatnam's Channel Dig-Out",
                subtitle: "READ ON MARITIME PROFESSIONAL",
                url: "http://www.maritimeprofessional.com/news/post-cyclone-dredging-underway-india-391883"
              },
              {
                quote: `"The closure of Chennai Ports resulted in a loss of over Rs. 100 crore per day."`,
                title: "₹100 Crore a Day: The Real Cost of a Storm-Shut Port",
                subtitle: "READ ON EASYVESSEL",
                url: "https://easyvessel.com/chennai-port-closure-cyclone-michaung-coastal-impact/"
              },
              {
                quote: `"Marine debris is increasing in the Lakshadweep as well, and it represents exactly how wide the footprint of the commercial fishery is."`,
                title: "1,150 Square Metres of Reef, Buried in Nets",
                subtitle: "READ ON MONGABAY INDIA",
                url: "https://india.mongabay.com/2021/01/ghosts-of-the-gulf-marine-debris-a-threat-to-corals-in-the-gulf-of-mannar/"
              },
              {
                quote: `"Fragmentation caused by these derelict nets impacted species such as Acropora, Pocillopora, and Porites."`,
                title: "Decades-Old Coral, Snapped Apart in Minutes",
                subtitle: "READ ON RESEARCHGATE",
                url: "https://www.researchgate.net/publication/356892712_Impact_of_Ghost_Nets_And_Marine_Debris_on_The_Coral_Reef_Ecosystem_of_Andrott_Island_Lakshadweep_India"
              },
              {
                quote: `"A dedicated team turned ghost-net 'busters' by policing the reefs for lost fishing gear."`,
                title: "Divers Turn Ghost-Net Busters to Save Tamil Nadu's Reefs",
                subtitle: "READ ON UNEP",
                url: "https://www.unep.org/news-and-stories/story/underwater-ghost-busting-save-indian-coral-reefs"
              }
            ].map((box, i) => (
              <a href={box.url} target="_blank" rel="noopener noreferrer" className="research-box" key={i}>
                <p className="research-quote">{box.quote}</p>
                <div className="research-author">
                  <div className="research-author-title">{box.title}</div>
                  <div className="research-author-subtitle" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    {box.subtitle} <span style={{ fontSize: '10px' }}>↗</span>
                  </div>
                </div>
              </a>
            ))}
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'center', marginTop: '4rem', paddingBottom: '4rem' }}>
          <PillNav
            items={[
              { label: 'CONTINUE', onClick: onContinue }
            ]}
            className="custom-nav"
            baseColor="#000000"
            pillColor="rgba(255,255,255,0.1)"
            hoveredPillTextColor="#ffffff"
            pillTextColor="#ffffff"
            initialLoadAnimation={false}
          />
        </div>
      </section>
    </div>
  );
}
