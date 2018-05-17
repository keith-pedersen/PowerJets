#include "ArrogantDetector.hpp"
#include <stdexcept>
#include <assert.h>

ArrogantDetector::~ArrogantDetector() {}

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
	// Similarly, ignore length, so we choose pT = 1
	return vec3_t(1., 0., std::sinh(eta));
}

double  ArrogantDetector::EtaMaxToPolarMax(double const etaMax_i) const
{
	double polarMax_i = kdp::RoundToNearestPitch(AbsPolarAngle(EtaToVec(etaMax_i)), settings.squareWidth);
	if(polarMax_i >= polarMax)
		polarMax_i -= settings.squareWidth; // Make sure there is a beam hole.
	assert(polarMax_i < polarMax);
		
	return polarMax_i;
}

void ArrogantDetector::Init_inDerivedCTOR()
{
	polarMax = AbsPolarAngle(vec3_t(0., 0., 1.));
		
	polarMax_cal = EtaMaxToPolarMax(settings.etaMax_cal);
	polarMax_track = EtaMaxToPolarMax(settings.etaMax_track);
	
	// Account for tiny little rounding errors near the edge of the detector.
	// These can cause (absPolarAngle < polarMax_cal) to return an invalid polarIndex
	{
		size_t count = 0;

		// Usually, one iteration through the loop is enough 
		while(GetPolarIndex(std::nextafter(polarMax_cal, 0.)) == NumTowerBands())
		{
			polarMax_cal = std::nextafter(polarMax_cal, 0.);
			
			if(++count > 10) throw std::runtime_error("ArrogantDetector: stuck in loop");
		}
	}
				GCC_IGNORE(-Wfloat-equal)
	// Account for the same rounding error when the tracker covers the whole detector
	if(settings.etaMax_track == settings.etaMax_cal)
		polarMax_track = polarMax_cal;
				GCC_IGNORE_END
		
	assert(GetPolarIndex(std::nextafter(polarMax_track, 0.)) < NumTowerBands());
		
	tooBigID = NumTowerBands() * settings.numPhiBins_centralBand;
}

void ArrogantDetector::SanityCheck_inDerivedCTOR()
{
	assert(polarMax_track <= polarMax_cal);
	
	towerID_t const polarIndex_CalEnd = NumTowerBands();
	
	// Ensure that polarMax_cal - epsilon has a valid polar index
	assert(GetPolarIndex(std::nextafter(polarMax_cal, 0.)) < polarIndex_CalEnd);
	
	// Ensure that phi is mapped properly
	for(towerID_t polarIndex = 0; polarIndex < polarIndex_CalEnd ; ++polarIndex)
	{
		assert(PhiWidth(polarIndex) >= settings.squareWidth);
		assert(NumPhiBins(polarIndex) <= settings.numPhiBins_centralBand);
		
		assert(GetPhiIndex(std::nextafter(M_PI, 0.), polarIndex) < settings.numPhiBins_centralBand);
	}
}

ArrogantDetector::towerID_t ArrogantDetector::NumTowerBands() const
{
	return towerID_t(std::round(polarMax_cal / settings.squareWidth));
}

ArrogantDetector::towerID_t ArrogantDetector::GetPolarIndex(double const absPolarAngle) const
{
	// These are asserts because these angles are determined by ArrogantDetector (and derived classes)
	// If they are wonky, we need to rewrite the detector code
	assert(absPolarAngle <= polarMax);
	assert(absPolarAngle >= 0.);
	
	return towerID_t(std::floor(absPolarAngle/settings.squareWidth));
}

ArrogantDetector::towerID_t ArrogantDetector::GetPhiIndex(double phi,
	towerID_t const polarIndex) const
{
	if(std::fabs(phi) > M_PI)
	{
		printf("%.16e\n", phi);
		assert(std::fabs(phi) <= M_PI);
	}
	
	// Because phi_min = -Pi, Pi can be an out of bounds index
	if(phi == M_PI) 
		phi = -M_PI;
	
	towerID_t const phiIndex = towerID_t(std::floor((phi + M_PI)/PhiWidth(polarIndex)));
	
	// When phi ~ Pi, we can get some rounding error. 
	// If we return a PhiIndex which is too large, it could create an invalid towerID.
	if(phiIndex == NumPhiBins(polarIndex))
	{
		//~ printf("%.3e\n", 1. - phi/M_PI);
		assert(1. - phi/M_PI < 4e-16);
		return phiIndex - 1;
	}
	else assert(phiIndex < NumPhiBins(polarIndex));
	
	return phiIndex;
}

std::pair<ArrogantDetector::towerID_t, ArrogantDetector::towerID_t>
ArrogantDetector::GetIndices_PolarPhi(double const absPolarAngle, double const phi) const
{
	towerID_t const polarIndex = GetPolarIndex(absPolarAngle);
	towerID_t const phiIndex = GetPhiIndex(phi, polarIndex);
	
	return std::pair<towerID_t, towerID_t>(polarIndex, phiIndex);
}

std::pair<ArrogantDetector::towerID_t, ArrogantDetector::towerID_t>
ArrogantDetector::GetIndices_PolarPhi(towerID_t const towerID) const
{
	assert(towerID < tooBigID); // b/c towerID_t is unsigned, this also ensures ID >= 0 (catching underflow)
	
	towerID_t const polarIndex = towerID / settings.numPhiBins_centralBand;
	towerID_t const phiIndex = towerID - polarIndex * settings.numPhiBins_centralBand;
	assert(phiIndex < settings.numPhiBins_centralBand);
	
	return std::pair<towerID_t, towerID_t>(polarIndex, phiIndex);
}

ArrogantDetector::towerID_t ArrogantDetector::GetTowerID(double const absPolarAngle, double const phi) const
{
	auto const indices = GetIndices_PolarPhi(absPolarAngle, phi);
	return (indices.first * settings.numPhiBins_centralBand) + indices.second;
}

std::array<double, 4> ArrogantDetector::GetTowerEdges(towerID_t const towerID) const
{
	auto const indices = GetIndices_PolarPhi(towerID);
	
	double const absTheta_left = ToAbsTheta(double(indices.first) * settings.squareWidth);
	double const absTheta_right = ToAbsTheta(double(indices.first + 1) * settings.squareWidth);
	assert(absTheta_left >= 0.);
	assert(absTheta_right > absTheta_left);
	assert(absTheta_right < M_PI); // ensuring beam hole
	
	double const deltaPhi = PhiWidth(indices.first);
	double const phi_left = -M_PI + deltaPhi * double(indices.second);
	assert(phi_left >= -M_PI);
	assert(phi_left < M_PI);
	
	return {absTheta_left, absTheta_right, phi_left, phi_left + deltaPhi};
}

ArrogantDetector::vec3_t ArrogantDetector::GetTowerCenter(towerID_t const towerID) const
{
	auto const edges = GetTowerEdges(towerID);
		
	double const absTheta = 	0.5*(edges[0] + edges[1]);
	double const phi = 			0.5*(edges[2] + edges[3]);
		
	double const cosTheta = std::cos(absTheta);
	return vec3_t(std::cos(phi) * cosTheta, std::sin(phi) * cosTheta, std::sin(absTheta));
}

std::vector<ArrogantDetector::vec3_t> ArrogantDetector::GetTowerArea() const
{
	std::vector<vec3_t> towerVec;
	
	/* REMEMBER: we are using a different polar angle: th = 0 --> equator, so cos(theta) is the 
	 * dOmega = \int cos(theta) dTheta dPhi = deltaPhi * (sin(t2) - sin(t1))
	 * 	= 2*sin((th2 - th1)/2)*cos((th1 + th2)/2)*dPhi
	 * 2 dPhi / (4 pi) = dPhi / (2 pi) = (2 pi)/numPhiBins / (2 pi) = 1./numPhiBins    so ....
	 * dOmega / (4 pi) = sin((th2 - th1)/2)*cos((th1 + th2)/2) / numPhiBins
	*/	
	towerID_t const polarIndex_end = NumTowerBands();
		
	for(towerID_t polarIndex = 0; polarIndex < polarIndex_end; ++polarIndex)
	{
		towerID_t const numPhiBins = towerID_t(std::round((2.*M_PI) / PhiWidth(polarIndex)));
				
		// NO shortcuts; theta and polarAngle do not neccesarily map linearly
		double const theta1 = ToAbsTheta(double(polarIndex) * settings.squareWidth);
		double const theta2 = ToAbsTheta(double(polarIndex + 1) * settings.squareWidth);
		
		// We assume that deltaPhi is the same over an entire band, true up to O(epsilon)
		double const dOmegaNorm = 
			std::sin(0.5*(theta2 - theta1))*std::cos(0.5*(theta1 + theta2)) / double(numPhiBins);
		
		// This is an inefficient way to get the tower center; however, it is the most readable.
		// And we don't expect this function to get called very often
		for(towerID_t phiIndex = 0; phiIndex < numPhiBins; ++phiIndex)
		{
			towerID_t const towerIndex = polarIndex * settings.numPhiBins_centralBand + phiIndex;
			towerVec.push_back(GetTowerCenter(towerIndex) * dOmegaNorm);
			towerVec.push_back(towerVec.back());
			towerVec.back().x3 = -towerVec.back().x3; // emplace the backward detector
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
		
		// not else if because ME may have photon, lepton
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

void ArrogantDetector::Clear()
{
	finalState.clear(); 
	tracks.clear();	
	towers.clear();
	
	if(clearME) 
		me.clear(); // Clear the matrix element, if requested
	clearME = true; // Clear next time. Only don't clear when called by operator(Pythia, ...)
}

void ArrogantDetector::operator()(std::vector<vec4_t> const& neutralVec,
	std::vector<vec4_t> const& chargedVec, std::vector<vec4_t> const& invisibleVec,
	std::vector<vec4_t> const& pileupVec)
{
	Clear();
	
	// These are only used in this function, so we clear here for clarity
	foreCal.clear(); backCal.clear(); 
		
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

//! @brief Return all visible energy in calorimeter towers, 
//  defined by the tower's edges <(theta_L, theta_R, phi_L, phi_R), energy>
std::vector<std::pair<std::array<double, 4>, double>> ArrogantDetector::AllVisibleAsTowers()
{
	// This should not have side effects
	foreCal.clear(); backCal.clear();
	
	std::vector<std::pair<std::array<double, 4>, double>> visible;
	
	std::vector<vec3_t> particles = tracks;
	particles.insert(particles.end(), towers.begin(), towers.end() - 1); // Don't copy the missing energy
	
	for(auto const& p3 : particles)
	{
		(IsForward(p3) ? foreCal : backCal)
			[GetTowerID(AbsPolarAngle(p3), p3.T().Phi())] += p3.Mag();
	}
	
	// We don't use this dOmega because it complicates the depiction on a plane
	// 2*sin((th2 - th1)/2)*cos((th1 + th2)/2)*dPhi
	
	for(auto itTower = foreCal.begin(); itTower not_eq foreCal.end(); ++itTower)
	{
		auto const edges = GetTowerEdges(itTower->first);
		double const dOmega = 1.;
		//~ 2.*std::sin(0.5*(edges[1]-edges[0]))*
			//~ std::cos(0.5*(edges[1]+edges[0])) * (edges[3]-edges[2]);
		
		visible.emplace_back(edges, itTower->second / dOmega);
	}
	
	for(auto itTower = backCal.begin(); itTower not_eq backCal.end(); ++itTower)
	{
		auto const edges = GetTowerEdges(itTower->first);
		double const dOmega = 1.;
		//~ 2.*std::sin(0.5*(edges[1]-edges[0]))*
			//~ std::cos(0.5*(edges[1]+edges[0])) * (edges[3]-edges[2]);
		
		visible.emplace_back(edges, itTower->second / dOmega);
		
		// Backward cal, reverse the theta (and swap positions, since smaller was larger)
		double const theta_L = -visible.back().first[1];
		visible.back().first[1] = -visible.back().first[0];
		visible.back().first[0] = theta_L;
	}

	return visible;
}

////////////////////////////////////////////////////////////////////////

ArrogantDetector_Lepton::~ArrogantDetector_Lepton() {}

double ArrogantDetector_Lepton::AbsPolarAngle(vec3_t const& particle) const
{
	return std::atan2(std::fabs(particle.x3), particle.T().Mag());
}

double ArrogantDetector_Lepton::PhiWidth
(towerID_t const thetaIndex) const
{
	return phiWidth[thetaIndex];
}

ArrogantDetector_Lepton::towerID_t ArrogantDetector_Lepton::NumPhiBins
(towerID_t const thetaIndex) const
{
	return numPhiBins[thetaIndex];
}

void ArrogantDetector_Lepton::Init_phiWidth()
{
	phiWidth.clear();
	numPhiBins.clear();
	
	// The number of phi towers depends on the central theta, 
	// due to the differential solid angle dOmega = cos(theta) dTheta dPhi
	// NOTE: this differential solid angle is different than standard polar angle, 
	// since theta = 0 corresponds to totally transverse
	
	// However, the real solid angle is \int cos(th) = dPhi * (sin(th2) - sin(th1))
	// dOmega = 2*sin((th2 -th1)/2)*cos((th1 + th2)/2)*dPhi
	// So for the central band, 
	// 	dOmega0 = 2*sin(sw/2)*cos(sw/2)*sw   (sw = squareWidth)
	// And for all other bands,
	// 	dOmega = 2*sin(sw/2)*cos((2th1 + sw)/2)*dPhi
	// Setting the two equal and solving for dPhi
	// 	dPhi = sw*cos(sw/2) / cos((2th1 + sw)/2)
	phiWidth.push_back(settings.squareWidth);
	numPhiBins.push_back(settings.numPhiBins_centralBand);
	
	double const dPhiNumerator = 
		settings.squareWidth * std::cos(0.5*settings.squareWidth);
		
	for(double theta1 = settings.squareWidth; 
		theta1 < polarMax_cal; theta1 += settings.squareWidth)
	{
		double dPhi = dPhiNumerator / std::cos(0.5*(2.*theta1 + settings.squareWidth));
		double const numPhiTowers = 2.*std::round(M_PI / dPhi);
		assert((size_t(numPhiTowers) bitand 1lu) == 0lu); // Check for even number of towers
				
		phiWidth.push_back((2. * M_PI) / numPhiTowers);
		numPhiBins.push_back(towerID_t(numPhiTowers));
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
