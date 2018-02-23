#include "ArrogantDetector.hpp"
#include <stdexcept>
#include <assert.h>

ArrogantDetector::towerID_t ArrogantDetector::Settings::Read_numPhiBins_central
(QSettings const& parsedINI, std::string const& detectorName, char const* const defaultValue)
{
	//~ return towerID_t(std::round((2.*M_PI)/kdp::ReadAngle<double>(parsedINI.
		//~ value((detectorName + "/squareWidth").c_str(), defaultValue).
			//~ toString().toStdString())));
	
	// Force an even number of phi bins
	return towerID_t(2.*std::round(M_PI/kdp::ReadAngle<double>(parsedINI.
		value((detectorName + "/squareWidth").c_str(), defaultValue).
			toString().toStdString())));
}
			
double ArrogantDetector::Settings::Read_double
(QSettings const& parsedINI, std::string const& detectorName, 
	std::string const& key, double const defaultVal)
{
	return parsedINI.value((detectorName + "/" + key).c_str(), defaultVal).toDouble();
}

////////////////////////////////////////////////////////////////////////

ArrogantDetector* ArrogantDetector::NewDetector(QSettings const& parsedINI, 
	std::string const& detectorName)
{
	static constexpr char const* key_lepton = "lep";
	static constexpr char const* key_hadron = "had";
	
	std::string type = parsedINI.value((detectorName + "/" + "type").c_str(), key_lepton).
		toString().toStdString();
		
	if(type.find(key_lepton, 0) not_eq std::string::npos)
		return new ArrogantDetector_Lepton(parsedINI, detectorName);
	else if (type.find(key_hadron, 0) not_eq std::string::npos)
		return new ArrogantDetector_Hadron(parsedINI, detectorName);
	else
		throw std::runtime_error("ArrogantDetector: only type containing \"lep\" of \"had\" is supported!");
}

ArrogantDetector::vec3_t ArrogantDetector::EtaToVec(double const eta)
{
	// theta = 2*atan(exp(-eta)) OR
	// p> = {pT cos(phi), pT sin(phi), pT sinh(eta)} = pT {cos(phi), sin(phi), sinh(eta)}
	// We can choose any phi, so make it simple ... phi = 0
	return vec3_t(1., 0., std::sinh(eta));
}

void ArrogantDetector::Init_inDerivedCTOR()
{
	polarMax = AbsPolarAngle(vec3_t(0., 0., 1.));
	polarMax_cal = kdp::RoundToNearestPitch(AbsPolarAngle(EtaToVec(settings.etaMax_cal)), settings.squareWidth);
	polarMax_track = kdp::RoundToNearestPitch(AbsPolarAngle(EtaToVec(settings.etaMax_track)), settings.squareWidth);

	tooBigID = towerID_t(std::round(polarMax_cal / settings.squareWidth)) * settings.numPhiBins_centralBand;
}

ArrogantDetector::towerID_t ArrogantDetector::GetTowerID(double const absPolarAngle, double const phi) const
{
	// These are asserts because these angles are determined by ArrogantDetector (and derived classes)
	// If they are wonky, we need to rewrite the detector code
	assert(absPolarAngle <= polarMax);
	assert(absPolarAngle >= 0.);
	assert(std::fabs(phi) <= M_PI);
	
	towerID_t const polarIndex = towerID_t(std::floor(absPolarAngle/settings.squareWidth));
	towerID_t const phiIndex = towerID_t(std::floor((phi + M_PI)/PhiWidth(polarIndex))); // phi_min = -pi
	return (settings.numPhiBins_centralBand * polarIndex) + phiIndex;
}

ArrogantDetector::vec3_t ArrogantDetector::GetTowerCenter(towerID_t const towerID) const
{
	assert(towerID < tooBigID); // b/c towerID_t is unsigned, this also ensures ID >= 0
	
	// The following asserts don't use >= / <= because a tower center should never reside on these exact boundaries
	
	towerID_t const polarIndex = (towerID / settings.numPhiBins_centralBand);
	double const absTheta = ToAbsTheta((double(polarIndex) + 0.5) * settings.squareWidth);
	assert(absTheta > 0.);
	assert(absTheta < M_PI);
	
	double const phi = -M_PI + PhiWidth(polarIndex) * 
		(double(towerID - (polarIndex * settings.numPhiBins_centralBand)) + 0.5);
	assert(phi > -M_PI);
	assert(phi < M_PI);
	
	double const cosTheta = std::cos(absTheta);
	return vec3_t(std::cos(phi) * cosTheta, std::sin(phi) * cosTheta, std::sin(absTheta));
}

std::vector<ArrogantDetector::vec3_t> ArrogantDetector::GetTowerArea() const
{
	std::vector<vec3_t> towerVec;
	
	/* dOmega = \int sin(theta) dTheta dPhi = deltaPhi * (cos(t2) - cos(t1))
	 * = deltaPhi * 2 sin(0.5(t1 + t2)) * sin(0.5(t2 - t1)) 
	 * 2 deltaPhi / (4 pi) = deltaPhi / (2 pi) = (2 pi)/numPhiBins / (2 pi)
	*/	
	towerID_t const polarIndex_end = towerID_t(std::round(polarMax_cal / settings.squareWidth));
		
	for(towerID_t polarIndex = 0; polarIndex < polarIndex_end; ++polarIndex)
	{
		towerID_t const numPhiBins = towerID_t(std::round((2.*M_PI) / PhiWidth(polarIndex)));
		
		// NO shortcuts; theta and polarAngle do not neccesarily map linearly
		double const theta1 = ToAbsTheta(polarIndex * settings.squareWidth);
		double const theta2 = ToAbsTheta(polarIndex * settings.squareWidth);
		
		double const dOmegaNorm = 
			std::sin(0.5*(theta1 + theta2))*std::sin(0.5*(theta2 - theta1)) / double(numPhiBins);
		
		// This is an inefficient way to get the tower center; however, it is the most readible.
		// And we don't expect this function to get called very often
		for(towerID_t phiIndex = 0; phiIndex < numPhiBins; ++phiIndex)
		{
			towerID_t towerIndex = polarIndex * settings.numPhiBins_centralBand + phiIndex;
			towerVec.push_back(GetTowerCenter(towerIndex) * dOmegaNorm);
		}
	}
	
	return towerVec;
}

// Fill the detector from the Pythia event
void ArrogantDetector::operator()(Pythia8::Pythia& theEvent, 
	std::vector<vec4_t> const& pileupVec)
{
	std::vector<vec4_t> neutralVec, chargedVec;
	std::vector<vec4_t> invisibleVec;
	
	me.clear();
	clearME = false; // Communicate to second call to operator() not to clear ME
		
	// Use a *signed* int as an index iterator because Pythia does (why?!!)
	for(int i = 0; i < theEvent.event.size(); ++i)
	{
		if(std::abs(theEvent.event[i].status()) == 23) // ME final state
			me.emplace_back(theEvent.event[i]);
		
		if(theEvent.event[i].isFinal())
		{
			Pythia8::Particle& particle = theEvent.event[i];
			
			kdp::Vec3 const p3(particle.px(), particle.py(), particle.pz());
				
			if(not particle.isVisible()) // i.e. invisible neutrinos
				invisibleVec.emplace_back(p3);
			else
				(particle.isCharged() ? chargedVec : neutralVec).
					emplace_back(particle.e(), p3, kdp::Vec4from2::Energy);
		}
	}
	
	(*this)(neutralVec, chargedVec, invisibleVec, pileupVec); // Do the actual detection
}

void ArrogantDetector::operator()(std::vector<vec4_t> const& neutralVec,
	std::vector<vec4_t> const& chargedVec, std::vector<vec4_t> const& invisibleVec,
	std::vector<vec4_t> const& pileupVec)
{
	foreCal.clear(); backCal.clear();

	finalState.clear(); tracks.clear();	towers.clear();
	
	if(clearME) me.clear(); // Clear the matrix element, if requested
	clearME = true; // Clear next time. Only don't clear when called by operator(Pythia, ...)
		
	finalStateE = visibleE = 0.;
	visibleP3 = vec3_t(0., 0., 0.);
	invisibleP3 = vec3_t(0., 0., 0.);
	
	/* Invisible doesn't show up in the detector,
	 * Pileup and track-subtracted excess doesn't show up in finalState.
	 * Thus we can create a single vector (allVec) which mereges all input,
	 * keeping the control logic simpler (i.e. no separate, redundant loops).
	 * Furthermore, the track subtraction scheme requires appending to neutralVec, 
	 * which requires creating a copy anyway.
	 *  
	 *           <neutP3 >
	 *           <--- finalState ---------->
	 * allVec = {invisible, charged, neutral, pileup, [track-subtracted excess]}
	 *                      <--- detected ------------------------------------>
	 *                      <trakd>
	 * 
	 * The track subtraction scheme will cause each tracked charged particle to 
	 * add a neutral after pileup. Since we will be using iterators to loop over allVec, 
	 * we must ensure that cnV has the capacity for all potential additions,
	 * so that all iterators remain valid. 
	*/	
		
	std::vector<vec4_t> allVec;
	allVec.reserve(invisibleVec.size() + 2*chargedVec.size()
		+ neutralVec.size() + pileupVec.size());
	
	allVec.insert(allVec.end(), invisibleVec.begin(), invisibleVec.end());
	
	auto const itChargedBegin = allVec.end();
	allVec.insert(allVec.end(), chargedVec.begin(), chargedVec.end()); 
	
	auto const itNeutralBegin = allVec.end();
	allVec.insert(allVec.end(), neutralVec.begin(), neutralVec.end());
	
	auto const itPileupBegin = allVec.end();
	allVec.insert(allVec.end(), pileupVec.begin(), pileupVec.end());
	
	// CARFEUL, we are appending to the vector we are iterating over
	for(auto itAll = allVec.begin(); itAll not_eq allVec.end(); ++itAll)
	{
		double const energy = itAll->x0;
		vec3_t const& p3 = itAll->p();
		
		if(itAll < itPileupBegin)
		{			
			// Here we make the mistake of assuming that everything is massless, 
			// so the finalState momentum probably won't balance 100%
			finalStateE += energy;
			//~ finalState.emplace_back(p3, energy, true); // true => p3 will be normalized when it is emplaced
			finalState.push_back(p3);
		}
		
		if(itAll < itChargedBegin) // fundamentally invisible energy
			invisibleP3 += p3;
		else // Everything else can potentially be seen in the detector
		{
			double const absPolarAngle = AbsPolarAngle(p3);
			assert(absPolarAngle >= 0.);
			
			if(absPolarAngle < polarMax_cal) // It's within the cal, the detector can see it
			{
				// We see a track when ... 
				if((itAll < itNeutralBegin) // it's charged
					and (absPolarAngle < polarMax_track) // within the tracker
					and (p3.T().Mag() > settings.minTrackPT)) // and has enough pT
				{
					// The detector will see the momentum of the charged particle and 
					// assume that it is massless, subtracting only it's momentum from 
					// the calorimeter cell. So we add the energy excess as a neutral particle.
					// We don't need to scale p3 in the new neutral because 
					// p3 is only used to determine direction.
					double const trackE = p3.Mag();
					
					visibleE += trackE;
					visibleP3 += p3;					
					
					// We're about to grow allVec; if it reallocates, all iterators become invalid
					assert(allVec.size() < allVec.capacity());
					
					//~ tracks.emplace_back(finalState.back().pHat, trackE, false); // Reuse pre-normalized vector
					//~ allVec.emplace_back(energy - trackE, p3, kdp::Vec4from2::Energy);
					
					tracks.push_back(p3);
					allVec.emplace_back(0., p3*(energy/trackE - 1.), kdp::Vec4from2::Mass);
				}
				else // we don't see a track, so it's seen by the calorimeter
				{
					visibleE += energy;
					
					(IsForward(p3) ? foreCal : backCal)
						[GetTowerID(absPolarAngle, p3.T().Phi())] += energy;
				}
			}
			// else it's not seen
		}
	}
		
	WriteCal(foreCal, false);
	WriteCal(backCal, true);
	
	AddMissingE();
	
	// Now that totalEnergy is known, normalize the energy fractions of tracks and finalState
	//~ for(PhatF& particle : finalState)
		//~ particle.f /= finalStateE;
		
	//~ for(PhatF& track : tracks)
		//~ track.f /= visibleE;
			
	//~ for(PhatF& tower : towers)
		//~ tower.f /= visibleE;
}

void ArrogantDetector::WriteCal(cal_t const& cal, bool const backward)
{
	// Convert all towers to massless 3-vectors
	for(auto itTower = cal.begin(); itTower not_eq cal.end(); ++itTower)
	{
		double const energy = itTower->second;
		// Emplace pre-normalized p3 (false means don't normalize)
		//~ towers.emplace_back(GetTowerCenter(itTower->first), energy, false);
		towers.push_back(GetTowerCenter(itTower->first)*energy);
		
		// Flip the z-coord for the backward detector
		if(backward)
			towers.back().x3 = -towers.back().x3;
		
		// Add the 3-momentum of the (massless) tower to the running sum
		//~ visibleP3 += (towers.back().pHat * energy);
		visibleP3 += towers.back();
	}
}

void ArrogantDetector::AddMissingE()
{
	double const missingE = visibleP3.Mag();
	visibleE += missingE;
	
	// MET is treated as a tower, due to poor angular resolution.
	//~ towers.emplace_back(-visibleP3, missingE);
	towers.push_back(-visibleP3);
}

////////////////////////////////////////////////////////////////////////

double ArrogantDetector_Lepton::AbsPolarAngle(vec3_t const& particle) const
{
	return std::atan2(std::fabs(particle.x3), particle.T().Mag());
}

double ArrogantDetector_Lepton::PhiWidth
(towerID_t const thetaIndex) const
{
	return phiWidth[thetaIndex];
}

void ArrogantDetector_Lepton::Init_phiWidth()
{
	phiWidth.clear();
		
	for(double thetaCenter = 0.5 * settings.squareWidth; 
		thetaCenter < polarMax_cal; thetaCenter += settings.squareWidth)
	{
		// The number of phi towers depends on the central theta, 
		// due to the differential solid angle dOmega = cos(theta) dTheta dPhi
		double const numPhiTowers = 
			2.*std::round((M_PI * std::cos(thetaCenter)) / settings.squareWidth);
		assert((size_t(numPhiTowers) bitand 1lu) == 0lu); // Check for even number of towers
		
		phiWidth.push_back((2. * M_PI) / numPhiTowers);
	}
}

////////////////////////////////////////////////////////////////////////

double ArrogantDetector_Hadron::AbsPolarAngle(vec3_t const& particle) const
{
	return std::fabs(particle.Eta());
}

// We have p> = {pT cos(phi), pT sin(phi), pT sinh(eta)}
// So theta = atan(sinh(eta)) (also known as the Gudermannian)
double ArrogantDetector_Hadron::ToAbsTheta(double const absEta) const
{
	return std::atan(std::sinh(absEta));
}

/*
ArrogantDetector_Lepton::towerID_t ArrogantDetector_Lepton::GetTowerID
	(real_t const absTheta, real_t const phi) const
{
	assert(absTheta >= 0.);
	assert(absTheta <= M_PI);
	assert(std::fabs(phi) <= M_PI);
	
	towerID_t const thetaIndex = towerID_t(std::floor(absTheta/settings.squareWidth));
	towerID_t const phiIndex = towerID_t(std::floor((phi + M_PI)/phiWidth[thetaIndex])); // phi_min = -pi
	return cumulativePhiBins[thetaIndex] + phiIndex; // old method
}
*/

/*
ArrogantDetector_Lepton::vec3_t ArrogantDetector_Lepton::GetTowerCenter(towerID_t const towerID)
{
	
	assert(towerID < cumulativePhiBins.back()); // b/c towerID_t is unsigned, this also ensures ID >= 0
	assert(cumulativePhiBins.front() == 0);
	
	if(towerID < *thetaBin)
	{
		// At the start of each cal write out, we reset to the beginning.
		thetaBin = cumulativePhiBins.cbegin();
	}
	
	while(towerID >= *(++thetaBin)); // increment till we find a cumulative total higher
	--thetaBin; // Then back up because we moved one theatBin too far.
	
	towerID_t const thetaIndex = towerID_t(thetaBin - cumulativePhiBins.cbegin());	
	real_t const theta = (real_t(thetaIndex) + 0.5) * settings.squareWidth;
	real_t const phi = ((real_t(towerID - *thetaBin) + 0.5) * phiWidth[thetaIndex]) - M_PI;
}
*/



//~ ArrogantDetector::ArrogantDetector(QSettings const& settings, std::string const& detectorName)
//~ {
	//~ // First get the requested squareWidth
	//~ {
		//~ std::string const fullKey = detectorName + squareWidth_key;
		
		//~ squareWidth = ReadAngle(settings, fullKey);
	//~ }
		
	//~ bool parseOK = true;
	//~ real_t numThetaBands_float;
	
	//~ // Next get the calTheta
	//~ {
		//~ std::string const fullKey = detectorName + calEta_key;
		
		//~ calTheta = EtaToTheta(settings.value(fullKey.c_str(), calEta_default).toDouble(&parseOK));
			
		//~ if(not parseOK)
			//~ throw std::runtime_error(("ArrogantDetector cannot parse: " + fullKey).c_str());
			
		//~ // Adjust squareWidth so it lines up with calTheta
		//~ numThetaBands_float = std::ceil((M_PI_2 - calTheta) / squareWidth);
		//~ squareWidth = (M_PI_2 - calTheta) / numThetaBands_float;
	//~ }
	
	//~ // Next get trackTheta
	//~ {
		//~ std::string const fullKey = detectorName + trackEta_key;

		//~ // Adjust so its an integer number of belts from the center
		//~ trackTheta = EtaToTheta(settings.value(fullKey.c_str(), trackEta_default).toDouble(&parseOK));
		//~ if(not parseOK)
			//~ throw std::runtime_error(("ArrogantDetector cannot parse: " + fullKey).c_str());
		
		//~ trackTheta = M_PI_2 - squareWidth * std::ceil((M_PI_2 - trackTheta) / squareWidth);
	//~ }
	
	//~ // Fill the forward cal
	//~ {
		//~ size_t const numThetaBands = size_t(numThetaBands_float);
		//~ foreCal.reserve(numThetaBands);
		//~ phiWidth.reserve(numThetaBands);
		//~ size_t numTowers = 0;
		
		//~ real_t thetaCenter = calTheta + 0.5*squareWidth;
		
		//~ for(size_t band = 0; band < numThetaBands; ++band)
		//~ {
			//~ // The number of phi towers depends on the central theta, 
			//~ // due to the differential solid angle dOmege = sin(theta) dTheta dPhi
			//~ size_t const numPhiTowers = 
				//~ size_t(std::ceil((2. * M_PI * std::sin(thetaCenter)) / squareWidth));
			//~ foreCal.emplace_back(numPhiTowers, 0.);
			//~ phiWidth.emplace_back(2.*M_PI / real_t(numPhiTowers));
			//~ numTowers += numPhiTowers;
			
			//~ thetaCenter += squareWidth;
		//~ }
		//~ assert((thetaCenter - squareWidth) < M_PI_2);
		
		//~ if(2*numTowers*sizeof(real_t) > byteLimit) // disregard negligible std::vector overhead
			//~ throw std::runtime_error("ArrogantDetector: too many towers, exceeds byte limit");
	//~ }
	
	//~ maxPhiTowers = calIndex_t(foreCal.back().size()); // This conversion ought to be fine, due to byteLimit
	//~ backCal = foreCal; // The backCal is a mirror image
//~ }
		
//~ // Fill the detector from the Pythia event
//~ // Validated 16 Mar 2017 by taking squareWidth very small,
//~ // MET disappears for those events lacking neutrinos
//~ void ArrogantDetector::operator()(Pythia8::Pythia& run)
//~ {
	//~ finalState.clear();
	//~ tracks.clear(); 				
	//~ towers.clear();
	//~ // The calorimeter and struckTowers have already been cleared
		
	//~ visibleP3 = vec3_t(); // Create a new, null running sum of visible 3-momentum
	//~ totalE = 0.;
	//~ real_t visibleE = 0.;

	//~ // Assume we're in an e+e- detector ... energy is known
	//~ for(int i = 0; i < run.event.size(); ++i)
	//~ {
		//~ if(run.event[i].isFinal())
		//~ {
			//~ Pythia8::Particle& particle = run.event[i];
			
			//~ // totalE is an easy way to get the collision energy without user input
			//~ real_t const energy = particle.e();
			//~ totalE += energy;
								
			//~ if(particle.isVisible()) // i.e. not neutrinos
			//~ {
				//~ kdp::Vec3 const p3(particle.px(), particle.py(), particle.pz());
				
				//~ // p3 will be automatically normalized when it is emplaced 
				//~ finalState.emplace_back(p3, energy);
				
				//~ // Get the polar angle to the closest beam
				//~ real_t const thetaPlus = ThetaPlus(p3);
				//~ assert(thetaPlus <= M_PI_2);
				
				//~ if(thetaPlus > calTheta) // Is it detectable?
				//~ {
					//~ visibleE += energy; // Then the calorimeter saw it
					
					//~ // Charged particles within the detection regime and
					//~ // above pT threshold are seen with perfect clarity.
					//~ // We assume perfect track-energy subtraction from the cal,
					//~ // and thus do not bin tracks in the cal
					//~ if(particle.isCharged() 
						//~ and (thetaPlus > trackTheta)
						//~ and (particle.pT() > minPT))
					//~ {
						//~ visibleP3 += p3;
						//~ tracks.push_back(finalState.back()); // Copy the finalState version
					//~ }
					//~ else
					//~ {
						//~ thetaCal_t& cal = IsForward(p3) ? foreCal : backCal;
						//~ std::set<calIndex_t>& struck = IsForward(p3) ? foreStruck : backStruck;
						
						//~ calIndex_t const thetaIndex = 
							//~ calIndex_t(std::floor((thetaPlus - calTheta)/squareWidth));
						//~ assert(thetaIndex < cal.size());
							
						//~ calIndex_t const phiIndex = 
							//~ calIndex_t(std::floor((p3.Phi() + M_PI) / phiWidth[thetaIndex]));
							//~ // p3.Phi is returned in (-pi, pi), from atan2
						//~ assert(phiIndex < cal[thetaIndex].size());
						//~ assert(phiIndex < maxPhiTowers);
						//~ //assert(size_t(std::round(2.*M_PI / phiWidth[thetaIndex])) == cal[thetaIndex].size());
						
						//~ cal[thetaIndex][phiIndex] += energy;
						//~ struck.insert(thetaIndex*maxPhiTowers + phiIndex);
					//~ }
				//~ }
			//~ }
		//~ }
	//~ }
	
	//~ // Now that totalEnergy is known, correct the energy fractions of tracks and finalState
	//~ for(PhatF& track : tracks) {track.f /= totalE;}
	//~ for(PhatF& particle : finalState) {particle.f /= totalE;}
	
	//~ WriteAndClearCal(foreCal, foreStruck, 1.);
	//~ WriteAndClearCal(backCal, backStruck, -1.);
	
	//~ // If there is missing energy, emplace it
	//~ // We can have visibleE == totalE, but visibleP3 != 0 because calorimeter approx position
	//~ if(visibleE not_eq totalE)
	//~ {
		//~ // Add the missing energy (with assumed energy fraction)
		//~ finalState.emplace_back(-visibleP3, 1. - visibleE/totalE);
		//~ // MET is best treated as a tower, since it's angular resolution is limited by tower resolution
		//~ towers.push_back(finalState.back());
	//~ }
	
	// Some validation code
	// Print out sum of visible energy, it's length compared to total energy, 
	// and the legitimate fraction of missing energy
	// The 4th col should approach the 5th col as squareWidth -> 0
	//~ printf("% .1e\t% .1e\t% .1e\t% .1e\t% .1e\n", 
		//~ visibleP3.x1, visibleP3.x2, visibleP3.x3, 
		//~ visibleP3.Mag()/totalE, 1. - visibleE/totalE);
		
	//~ for(auto thetaCal : foreCal)
	//~ {
		//~ for(auto phiCal : thetaCal)
		//~ {
			//~ assert(phiCal == 0.);
		//~ }
	//~ }
		
	//~ for(auto thetaCal : backCal)
	//~ {
		//~ for(auto phiCal : thetaCal)
		//~ {
			//~ assert(phiCal == 0.);
		//~ }
	//~ }
//~ }
	
//~ void ArrogantDetector::WriteAndClearCal
	//~ (thetaCal_t& cal, std::set<calIndex_t>& struck, real_t const zSign)
//~ {
	//~ // Convert all towers to massless 3-vectors
	//~ for(auto itTowerIndex = struck.begin(); itTowerIndex not_eq struck.end(); ++itTowerIndex)
	//~ {
		//~ calIndex_t const thetaIndex = (*itTowerIndex) / maxPhiTowers;
		//~ calIndex_t const phiIndex = (*itTowerIndex) - thetaIndex * maxPhiTowers;
		
		//~ real_t& energy = cal[thetaIndex][phiIndex];
		//~ {
			//~ real_t const theta = (real_t(thetaIndex) + 0.5)*squareWidth + calTheta;
			//~ real_t const phi = (real_t(phiIndex) + 0.5)*phiWidth[thetaIndex] - M_PI;
			
			//~ // Emplace pre-normalized with correct energy fraction (false means don't normalize)
			//~ towers.emplace_back(vec3_t(1., theta, phi, kdp::Vec3_from::LengthThetaPhi), 
				//~ energy/totalE, false);
		//~ }
		
		//~ // Correct the z-coord for the forward/backward detector
		//~ towers.back().pHat.x3 *= zSign;
		
		//~ // Add the 3-momentum of the tower to the running sum
		//~ visibleP3 += (towers.back().pHat*energy);
		//~ energy = 0.; // Clear the cal now, when we know the bin
		//~ // Given the cal's non-regular form, this is much easier than a large assign
	//~ }
	//~ // Now that the cal is clear, we can clear the list of struck towers
	//~ struck.clear();
//~ }
