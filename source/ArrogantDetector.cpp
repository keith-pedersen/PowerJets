// Copyright (C) 2018 by Keith Pedersen (Keith.David.Pedersen@gmail.com)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ArrogantDetector.hpp"
#include "kdp/kdpVectors.hpp"
#include <stdexcept>
#include <algorithm> // is_sorted, upper_bound
#include <assert.h>

////////////////////////////////////////////////////////////////////////
// ArrogantDetector::Settings
////////////////////////////////////////////////////////////////////////

ArrogantDetector::Settings::Settings(QSettings const& parsedINI, std::string const& detectorName)
{
	// Read in the requested squareWidth, which is the phi width of towers in the central band
	squareWidth = kdp::ReadAngle<double>(
		squareWidth.ReadVariant(parsedINI, detectorName).toString().toStdString());
		
	evenTowers.Read(parsedINI, detectorName);
		
	try
	{
		// Now round squareWidth to fit around the circle
		squareWidth = PhiSpec(squareWidth, evenTowers).DeltaPhi();
	}
	catch(std::domain_error e)
	{
		throw std::runtime_error("ArrogantDetector::Settings: squareWidth ("
			+ std::to_string(squareWidth) + ") is too small.");
	}
	
	etaMax_cal.Read(parsedINI, detectorName);
	etaMax_track.Read(parsedINI, detectorName);
	
	if(etaMax_track > etaMax_cal)
		throw std::runtime_error("ArrogantDetector: cannot track beyond edge of calorimeter (i.e. must find etaMax_track <= etaMax_cal).");
	
	minTrackPT.Read(parsedINI, detectorName);
}

////////////////////////////////////////////////////////////////////////

//~ double ArrogantDetector::Settings::Read_double
//~ (QSettings const& parsedINI, std::string const& detectorName, 
	//~ std::string const& key, double const defaultVal)
//~ {
	//~ return parsedINI.value((detectorName + "/" + key).c_str(), defaultVal).toDouble();
//~ }

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

ArrogantDetector::RawTower::RawTower(Edges&& edges, double const energy_in):
	Edges(edges), energy(energy_in) {}

////////////////////////////////////////////////////////////////////////	
	
void ArrogantDetector::RawTower::FlipZ()
{
	// We must keep lower < upper when we make eta opposite
	std::swap(eta_lower, eta_upper);
	eta_lower = -eta_lower;
	eta_upper = -eta_upper;
}
			
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

ArrogantDetector::PhiSpec::PhiSpec(double const deltaPhi_target, bool forceEven)
{
	// forceEven tells us whether to double before or after rounding
	double const numTowers_f = forceEven ? 
		2.*std::max(1., std::round(M_PI / deltaPhi_target)) : 
		std::max(1., std::round(2.*(M_PI / deltaPhi_target)));
	
	// The set of natural numbers which are contiguously representable 
	// with floating point is [0, 2^P]. 
	// Given C++ definition of epsilon, 2^P == 2 / epsilon
	// We must ensure that numTowers_f is within this set, 
	// otherwise we have a deltaPhi which is too small for exact rounding
	// (and just too small in general; that's just too damn small)
	if(numTowers_f > 2./std::numeric_limits<double>::epsilon())
		throw std::domain_error("PhiSpec: deltaPhi_target (" 
			+ std::to_string(deltaPhi_target) + ") is too small.");
		
	deltaPhi = 2.*(M_PI / numTowers_f);
	numTowers = towerID_t(numTowers_f);
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

ArrogantDetector::ArrogantDetector(QSettings const& parsedINI,
	std::string const& detectorName):
settings(parsedINI, detectorName), 
numBelts(0), tooBigID(0), equalArea(false), finalized(true) {}
		
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

///////////////////////////////////////////////////////////////////////

size_t ArrogantDetector::NumTowers() const
{
	size_t numTowers = 0;
	
	// Add up all the towers in all the belts
	for(size_t beltIndex = 0; beltIndex < numBelts; ++beltIndex)
		numTowers += DeltaPhi(beltIndex).NumTowers();
		
	return 2*numTowers; // We counted forward detector, which the backward detector mirrors
}

///////////////////////////////////////////////////////////////////////

PowerSpectrum::DetectorObservation ArrogantDetector::GetObservation(double const beta_longitudenalBoost) const
{
	if(not finalized)
		throw std::runtime_error("ArrogantDetector: detector not finalzed (you probably didn't mean to do this).");
	
															GCC_IGNORE_PUSH(-Wfloat-equal)
	// If there is no boost, return the lab frame observation
	if(beta_longitudenalBoost == 0.)
		return DetectorObservation(tracks, towers, towerAreas);
															GCC_IGNORE_POP
															
	// No "else" because return is obvious, and we want to limit scope of -Wfloat-equal ignore
	DetectorObservation observation;
	kdp::LorentzBoost<double> boost(vec3_t(0., 0., beta_longitudenalBoost));	
	
	// A PhatF will be constructed with a normalized pHat, and f = vec3.Mag();
	// we will normalize after every particle is in place.
	
	for(auto const& track : tracks)
		observation.tracks.emplace_back(boost.Forward(track));
	
	if(equalArea)
		observation.towerAreas.push_back(meanArea.Mean());
		
	for(auto const& tower : towers)
	{
		vec3_t const boostedTower = boost.Forward(tower);
		
		observation.towers.emplace_back(boostedTower);
				
		if(not equalArea)
		{
			/* When we boost towers, we need to adjust their fractional area. 
			 * If we boost the edges of the tower, it will no longer be square, 
			 * and so a circular cap will no longer be a good approximation.
			 * However, instead of boosting the detected objects, we can imagine
			 * boosting the detector, and asking which part of the detector 
			 * towers WOULD HAVE struck. Forward towers boosted backward 
			 * would have struck a more central area, and been measured
			 * with a CENTRAL angular resolution. Central towers boosted backwards
			 * would have been measured with FORWARD angular resolution. 
			 * 
			 * This scheme makes sense in both tower schemes (equal-area and square).
			 * In equal area, the fractional area is the same everywhere. 
			 * But central towers boosted forward will bunch up, 
			 * yet all with the same poor angular resolution, 
			 * so they're like a larger, low-res tower.
			 * In the square tower scheme, deltaPhi is the same in every belt, 
			 * and deltaEta is constant for all belts. Thus, towers in adjacent 
			 * eta belts will remain in adjacent eta belts, 
			 * just as if we had measured them with a smaller angular resolution.
			 */
			
			// throw a runtime_error if we fuck up the belt_index lookup
			double const boosted_surfaceFraction = surfaceFraction.at(
				// if the new belt is beyond the range of the detector, 
				// use the angular resolution of the last belt
				std::min(numBelts - 1,
					GetBeltIndex_AbsTanTheta(ArrogantDetector::AbsTanTheta(boostedTower))));
			
			observation.towerAreas.push_back(boosted_surfaceFraction);
		}
	}
	
	observation.NormalizeF();
	observation.CheckValidity_TowerArea();	
	return observation;
}

///////////////////////////////////////////////////////////////////////

// This function makes a lot of assumptions about how things are supposed to be.
// It assumes that the author of a derived class has read all the documentation
// about the responsibilities of a derived class (which are not long).
void ArrogantDetector::Init_inDerivedCTOR()
{
				GCC_IGNORE_PUSH(-Wfloat-equal)
	/////////////////////////////////////////////////////////////////////	
	// Sanity checks on beltEdges_eta
	
	// It should be filled ...
	if(beltEdges_eta.empty())
		throw std::runtime_error("ArrogantDetector: No calorimeter possible with supplied parameters (or derived ctor did not setup).");
	
	// ... and should be in the correct format (lower edges + last upper edge, sorted, unique)
	// any other format is a bug that needs repairing 
	// (i.e. the derived ctor should be passing only the correct format, 
	// and it has the discretion to issue a runtime_error if something doesn't make sense).
	assert(beltEdges_eta.front() == 0.);
	assert(std::is_sorted(beltEdges_eta.cbegin(), beltEdges_eta.cend()));
	assert(std::adjacent_find(beltEdges_eta.begin(), beltEdges_eta.end()) == beltEdges_eta.end());
	assert(beltEdges_eta.back() <= settings.etaMax_cal);
		
	/////////////////////////////////////////////////////////////////////
	// The last beltEdge is the uper-bound on the last belt (etaMax_cal)
	numBelts = beltEdges_eta.size() - 1; // so the numBelts is one less than size
	tooBigID = numBelts * maxTowersPerBelt;
	
	/////////////////////////////////////////////////////////////////////
	// Sanity checks on etaMax, PhiWidth and NumPhiBins
	assert(settings.etaMax_track <= settings.etaMax_cal); // Should be enforced by Settings ctor
	
	// The control logic requires a beam hole; a detector without one is no remotely possible.
	if(std::isinf(settings.etaMax_cal))
		throw std::runtime_error("ArrogantDetector: etaBins MUST have a beam hole (use finalState for truth level information).");
	
	for(towerID_t beltIndex = 0; beltIndex < numBelts; ++beltIndex)
	{
		PhiSpec const& belt = DeltaPhi(beltIndex);
		
		assert(belt.DeltaPhi() > 0.);
		assert(belt.NumTowers() > 1);
		assert(belt.NumTowers() <= maxTowersPerBelt);
	}
	
	/////////////////////////////////////////////////////////////////////	
	// Now map eta to tanTheta, for use in PartialFill(vectors ...)
	// Since PartialFill will determine positions from tanTheta, 
	// we should map our tanTheta back to eta so there's a 
	// perfect floating point correspondence between eta and tanTheta.
	// NOTE: reserving vector size is pointless; they aren't large enough or filled often enough
	
	for(double& eta : beltEdges_eta)
	{
		beltEdges_tanTheta.push_back(Eta_to_TanTheta(eta));
		eta = TanTheta_to_Eta(beltEdges_tanTheta.back());
	}
	assert(beltEdges_tanTheta.size() == beltEdges_eta.size());
	
	tanThetaMax_cal = beltEdges_tanTheta.back();
	settings.etaMax_cal = TanTheta_to_Eta(tanThetaMax_cal);
	
	/////////////////////////////////////////////////////////////////////
	// Also double-map the maximum eta for tracks
	settings.etaMax_track = TanTheta_to_Eta(Eta_to_TanTheta(settings.etaMax_track));
	
	if(settings.etaMax_track > settings.etaMax_cal) // Correct rounding error
		settings.etaMax_track = settings.etaMax_cal;
	
	/////////////////////////////////////////////////////////////////////
	// Set tanThetaMax_track
	// Only track the whole calorimeter when the two etaMax are equal
	if(settings.etaMax_track == settings.etaMax_cal)
		tanThetaMax_track = tanThetaMax_cal;
	else // Shift tracking to the belt whose upper edge is smaller than etaMax_track
	{
		// upper_bound finds the first edge greater than etaMax_track, 
		// so we subtract one to find the first edge less than
		towerID_t edgeIndex = 
			towerID_t(std::upper_bound(beltEdges_eta.begin(), beltEdges_eta.end(), 
				settings.etaMax_track) - beltEdges_eta.begin()) - 1;
			
		assert(edgeIndex < numBelts);
		
		// They two values are at the same index
		settings.etaMax_track = beltEdges_eta[edgeIndex];
		tanThetaMax_track = beltEdges_tanTheta[edgeIndex];
	}
	assert(tanThetaMax_track <= tanThetaMax_cal);	
	
	/////////////////////////////////////////////////////////////////////	
	// Now for each belt, calculate the fractionalSolidAngle (same for each tower)
	// and the prototype (phi=0) tower vec3 (simply rotate the
	// transverse component to each tower's actual phi).
	// Math worked out in TowerCenter.nb
	double theta_lower = 0.;
	
	for(towerID_t belt_iPlus1 = 1; belt_iPlus1 < beltEdges_tanTheta.size(); ++belt_iPlus1)
	{
		double const theta_upper = std::atan(beltEdges_tanTheta[belt_iPlus1]);
		
		{
			double const deltaTheta = theta_upper - theta_lower;
			double const thetaSum = theta_lower + theta_upper;
			double const deltaPhi = DeltaPhi(belt_iPlus1-1);
			
			double const solidAngle = 2. * deltaPhi * 
				(std::cos(0.5 * thetaSum) * std::sin(0.5 * deltaTheta));
			
			surfaceFraction.emplace_back(0.25*(solidAngle / M_PI));
			meanArea += surfaceFraction.back();
				
			towerPrototype.emplace_back(false); // don't initialize
			vec3_t& prototype = towerPrototype.back();
		
			double const sinDeltaTheta = std::sin(deltaTheta);
		
			/// Transverse term
			prototype.x2 = prototype.x1 = 
				std::sin(0.5 * deltaPhi) * 
				(deltaTheta + std::cos(thetaSum)*sinDeltaTheta);
			// Longitudinal term
			prototype.x3 = 0.5 * deltaPhi * sinDeltaTheta * std::sin(thetaSum);
			
			// Normalize length (which is currently proportional to solid angle). 
			// While Normalize() would better account for rounding errors, 
			// until we multiply transverse term by {cos(phi), sin(phi)}, the vector is too long.
			prototype /= solidAngle;
		}
		
		theta_lower = theta_upper; // Move to next belt
	}	
			GCC_IGNORE_POP // -Wfloat-equal
			
	// Now determine if this is an equal area detector
	if(std::sqrt(meanArea.Variance_Unbiased())/meanArea.Mean() < 1e-3)
	{
		equalArea = true;
	}
	
	Clear();
}

////////////////////////////////////////////////////////////////////////

ArrogantDetector::TowerIndices::TowerIndices(TowerID towerID):
	beltIndex(towerID/maxTowersPerBelt), phiIndex(towerID - beltIndex*maxTowersPerBelt) {}
	
////////////////////////////////////////////////////////////////////////

ArrogantDetector::TowerIndices::operator TowerID() const
{
	return TowerID(beltIndex*maxTowersPerBelt + phiIndex);
}

////////////////////////////////////////////////////////////////////////

ArrogantDetector::towerID_t ArrogantDetector::GetBeltIndex_AbsTanTheta(double const absTanTheta) const
{
	assert(absTanTheta >= 0.);
	//~ assert(absTanTheta <= tanThetaMax_cal); // sometimes it will be out of bounds
	
	// upper_bound finds the first edge greater than, so we subtract one to get the lower edge
	// if no edge is greater, it can't be detected, and we will return numBins
	towerID_t const beltIndex = (std::upper_bound(beltEdges_tanTheta.begin(), 
		beltEdges_tanTheta.end(), absTanTheta) - beltEdges_tanTheta.begin()) - 1;
		
	//~ assert(beltIndex < numBelts); // sometimes it will be out of bounds
	return beltIndex;
}

////////////////////////////////////////////////////////////////////////

ArrogantDetector::towerID_t ArrogantDetector::GetPhiIndex(double phi, towerID_t const beltIndex) const
{
	assert(beltIndex < numBelts);
	auto const& phiSpecs = DeltaPhi(beltIndex);
	
	towerID_t phiIndex = towerID_t(std::floor((phi + M_PI)/phiSpecs.DeltaPhi()));
	
	// When phi ~ Pi, we can get some rounding error. If we return a too-big piIndex,
	// it could create a towerID which is technically invalid.
	assert(phiIndex <= phiSpecs.NumTowers());
	if(phiIndex == phiSpecs.NumTowers())
		phiIndex = 0;
		
	return phiIndex;
}

////////////////////////////////////////////////////////////////////////

ArrogantDetector::TowerIndices ArrogantDetector::GetIndices_AbsTanTheta_Phi(double const absTanTheta, 
	double const phi) const
{
	towerID_t const beltIndex = GetBeltIndex_AbsTanTheta(absTanTheta);
	towerID_t const phiIndex = GetPhiIndex(phi, beltIndex);
	
	return TowerIndices(beltIndex, phiIndex);
}

////////////////////////////////////////////////////////////////////////

double ArrogantDetector::GetCentralPhi(TowerIndices const& indices) const
{
	auto const& phiSpecs = DeltaPhi(indices.beltIndex);
	assert(indices.phiIndex < phiSpecs.NumTowers());
	
	return -M_PI + (double(indices.phiIndex) + 0.5) * DeltaPhi(indices.beltIndex);
}

////////////////////////////////////////////////////////////////////////

ArrogantDetector::vec3_t ArrogantDetector::GetTowerCenter(TowerIndices const& indices) const
{
	vec3_t center(towerPrototype.at(indices.beltIndex));
		
	double const phi = GetCentralPhi(indices); // This will check for valid phiIndex
	// Rotate the prototype to the actual phi position
	center.x1 *= std::cos(phi);
	center.x2 *= std::sin(phi);
	
	return center;
}

ArrogantDetector::Edges ArrogantDetector::GetTowerEdges(TowerIndices const& indices) const
{
	assert(indices.beltIndex < numBelts);
	auto const& phiSpecs = DeltaPhi(indices.beltIndex);
	assert(indices.phiIndex < phiSpecs.NumTowers());
	
	double const phi_lower = -M_PI + phiSpecs.DeltaPhi() * double(indices.phiIndex);
	double const phi_upper = std::min(M_PI, phi_lower + phiSpecs.DeltaPhi()); // correct small rounding error, if present
	
	return Edges{beltEdges_eta[indices.beltIndex], beltEdges_eta[indices.beltIndex + 1], 
		phi_lower, phi_upper};
}

////////////////////////////////////////////////////////////////////////

void ArrogantDetector::Clear()
{
	me.clear();
	finalState.clear();
	tracks.clear();
	tracks_PU.clear();
	towers.clear();
	towerAreas.clear();
	
	visibleE = pileupE = 0.;
	visibleP3 = vec3_t(0., 0., 0.);
	invisibleP4 = vec4_t(0., 0., 0., 0.);
	
	foreCal.clear();
	backCal.clear();	
	
	finalized = false;
}

////////////////////////////////////////////////////////////////////////

void ArrogantDetector::Finalize(METcorrection const method)
{
	if(finalized)
		throw std::runtime_error("ArrogantDetector: Detector already finalized; you probably didn't mean to do this.");
	
	WriteCal(foreCal, false);
		foreCal.clear();
	WriteCal(backCal, true); // flipZ = true
		backCal.clear();
	
	AddMissingE(method);
	
	finalized = true;
}

////////////////////////////////////////////////////////////////////////

void ArrogantDetector::PartialFill(std::vector<vec4_t> const& neutralVec,
	std::vector<vec4_t> const& chargedVec, std::vector<vec4_t> const& invisibleVec, 
	std::vector<vec4_t> const& neutralPileup, std::vector<vec4_t> const& chargedPileup)
{
	/* Invisible doesn't show up in the detector,
	 * Pileup and track-subtracted excess doesn't show up in finalState.
	 * Thus we can create a single vector (allVec) which merges all input,
	 * keeping the control logic simpler (i.e. no separate, redundant loops).
	 * Furthermore, the track subtraction scheme requires appending to neutralVec, 
	 * which requires creating a copy anyway.
	 *  
	 *           <invblp3>
	 *           <--- finalState ---------->
	 * allVec = {invisible, neutral, charged, pileup^+, pileup^0, [track-subtracted excess]}
	 *                      <--- detected ------------------------------------------------>
	 *                               <--- tracked --->
	 * 
	 * The track subtraction scheme will cause each tracked charged particle to 
	 * add a neutral after pileup. Since we will be using iterators to loop over allVec, 
	 * we must ensure that cnV has the capacity for all potential additions,
	 * so that all iterators remain valid. 
	*/
	
	// TODO: using iterators here is really unnecessary. The time is not spent 
	// looking up values. It would be much safer and more readable to use indices.
	
	// MUST reserve space for all track excess, because we are growing list in place
	// and using iterators to refer to positions. 
	std::vector<vec4_t> allVec;
	allVec.reserve(invisibleVec.size() + 2*(chargedVec.size() + chargedPileup.size())
		+ neutralVec.size() + neutralPileup.size());
		
	size_t const capacity = allVec.capacity();
	
	allVec.insert(allVec.end(), invisibleVec.begin(), invisibleVec.end());
	auto const invisible_end = allVec.end();
	
	allVec.insert(allVec.end(), neutralVec.begin(), neutralVec.end());
	auto const charged_beg = allVec.end();
	
	allVec.insert(allVec.end(), chargedVec.begin(), chargedVec.end());
	auto const finalState_end = allVec.end();
	
	allVec.insert(allVec.end(), chargedPileup.begin(), chargedPileup.end());
	auto const charged_end = allVec.end();
	
	allVec.insert(allVec.end(), neutralPileup.begin(), neutralPileup.end());

	// Ensure that all iterators are still valid.
	assert(capacity == allVec.capacity());
		
	// CARFEUL, we are appending to the vector we are iterating over
	for(auto itAll = allVec.begin(); itAll not_eq allVec.end(); ++itAll)
	{
		double const energy = itAll->x0;
		vec3_t const& p3 = itAll->p();
		double const pMag = p3.Mag();
		
		if(energy < 0.)
			throw std::runtime_error("ArrogantDetector: Negative energy.");
		if(pMag > energy*(1. + 1e-8)) // Add a little safety buffer for light-like particles
			throw std::runtime_error("ArrogantDetector: Spacelike-particle: [" + 
				std::to_string(energy) + ", " + std::to_string(pMag) + "]");
		if(pMag > 0.) // Only particles that move are detected;
		// this ensures that we get a meaningful TowerID for neutral particles
		{
			// finalState is the vector of everything that came from the hard scatter
			// (including neutrinos). We are not interested in pileup.
			if(itAll < finalState_end)
				finalState.push_back(*itAll);
									
			if(itAll < invisible_end) // fundamentally invisible energy
				invisibleP4 += *itAll;
			else // Everything else can potentially be seen in the detector
			{
				double const absTanTheta = AbsTanTheta(p3);
				
				if(absTanTheta < tanThetaMax_cal) // It's within the cal, the detector can see it
				{
					// We see a track when ... 
					if(((itAll >= charged_beg) and (itAll < charged_end)) // it's charged
						and (absTanTheta < tanThetaMax_track) // within the tracker
						and (p3.T().Mag() > settings.minTrackPT)) // and has enough pT
					{
						// The detector will see the momentum of the charged particle and 
						// assume that it is massless, subtracting only it's momentum from 
						// the calorimeter cell. So we add the energy excess as a neutral particle.
						double const& trackE = pMag; // hopefully compiler will ellude
						
						visibleE += trackE;
						visibleP3 += p3;
						
						// Bin excess energy when there is some; 
						// otherwise we create a null 3-vector, which screws up the calorimeter lookup
						if(trackE < energy)
						{
							// We're about to grow allVec; if it reallocates, all iterators become invalid
							assert(allVec.size() < allVec.capacity());
							
							allVec.emplace_back(0., p3*((energy - trackE)/trackE), kdp::Vec4from2::Mass);
							
							// THIS IS NOT SAFE; it will round to zero when trackE ~= energy
							//~ allVec.emplace_back(0., p3*(energy/trackE - 1.), kdp::Vec4from2::Mass);
						}
						
						if(itAll < finalState_end)
							tracks.push_back(p3);
						else
						{
							pileupE += trackE;
							tracks_PU.push_back(p3);
						}
					}
					else // we don't see a track, so it's seen by the calorimeter
					{
						visibleE += energy;
						
						(IsForward(p3) ? foreCal : backCal)
							[GetIndices_AbsTanTheta_Phi(absTanTheta, p3.T().Phi())] += energy;
					}
				}
				// END: inside the cal
			}// END: visible 
		}// END: moving
	}// END: allVec loop
}

////////////////////////////////////////////////////////////////////////

void ArrogantDetector::PartialFill(Pythia8::Pythia& theEvent, bool const isPileup)
{
	std::vector<vec4_t> neutralVec, chargedVec, invisibleVec;
	
	if(not isPileup)
		me.clear();
		
	// Use a *signed* int as an index iterator because Pythia does (why?!!)
	for(int i = 0; i < theEvent.event.size(); ++i)
	{
		if((not isPileup) and (std::abs(theEvent.event[i].status()) == 23)) // ME final state
			me.emplace_back(theEvent.event[i]);
		
		// remember ... ME may have observable particle (photon, lepton, ...), 
		// so check for final state even if it has status 23
		if(theEvent.event[i].isFinal())
		{
			Pythia8::Particle& particle = theEvent.event[i];
			
			kdp::Vec3 const p3(particle.px(), particle.py(), particle.pz());
				
			if(not particle.isVisible()) // i.e. invisible neutrinos
				invisibleVec.emplace_back(p3);
			else
			{
				(particle.isCharged() ? chargedVec : neutralVec).
					emplace_back(particle.e(), p3, kdp::Vec4from2::Energy);
			}
		}
	}
	
	if(isPileup)
		PartialFill(std::vector<vec4_t>(), std::vector<vec4_t>(), 
			invisibleVec, neutralVec, chargedVec);
	else
		PartialFill(neutralVec, chargedVec, invisibleVec);
}
			
////////////////////////////////////////////////////////////////////////

void ArrogantDetector::operator()(std::vector<vec4_t> const& neutralVec,
	std::vector<vec4_t> const& chargedVec, std::vector<vec4_t> const& invisibleVec,
	std::vector<vec4_t> const& neutralPileup, std::vector<vec4_t> const& chargedPileup)
{
	Clear();	
		PartialFill(neutralVec, chargedVec, invisibleVec, neutralPileup, chargedPileup);	
	Finalize();
}

////////////////////////////////////////////////////////////////////////

void ArrogantDetector::operator()(Pythia8::Pythia& theEvent)
{
	Clear();
		PartialFill(theEvent, false); // isPileup = false
	Finalize();
}

////////////////////////////////////////////////////////////////////////

void ArrogantDetector::WriteCal(cal_t const& cal, bool const flipZ)
{
	// Convert all towers to massless 3-vectors
	for(auto itTower = cal.begin(); itTower not_eq cal.end(); ++itTower)
	{
		{
			TowerIndices twrIndices(itTower->first); // first == towerID
			
			towers.push_back(GetTowerCenter(twrIndices));
			
			if(not equalArea)
				towerAreas.push_back(surfaceFraction.at(twrIndices.beltIndex));
		}
		vec3_t& p3 = towers.back();
		
		p3 *= itTower->second; // second == tower energy
		
		// Flip the z-coord for the backward detector
		if(flipZ)
			p3.x3 = -p3.x3;
		
		// Add the 3-momentum of the (massless) tower to the running sum
		visibleP3 += p3;
	}
	
	if(equalArea and towers.size())
		towerAreas.assign(1, meanArea.Mean());
}

////////////////////////////////////////////////////////////////////////F

// TODO: This is probably not the correct approach to missing energy
void ArrogantDetector::AddMissingE(METcorrection const method)
{
	switch(method)
	{
		case METcorrection::None:
		break;
		
		case METcorrection::ToAllTowers:
		{
			// Discard all tracked pileup
			for(vec3_t const& pileup : tracks_PU)
			{
				visibleP3 -= pileup;
				visibleE -= pileup.Mag();
			}
			
			vec3_t const correctionP3 = -visibleP3;
				
			// MET is treated as a tower, due to poor angular resolution, 
			// but it is not correct to simply add it as if it was detected.
			// Instead, we spread it out proportional to each tower's fraction of the tower energy.
			double const towerE = std::accumulate(towers.begin(), towers.end(), 0., 
				[](double const sum, vec3_t const& tower){return sum + tower.Mag();});
			
			for(vec3_t& tower : towers)
				tower += correctionP3 * (tower.Mag() / towerE);
		}
		break;
	}	
}

////////////////////////////////////////////////////////////////////////

std::vector<ArrogantDetector::RawTower> ArrogantDetector::GetAllTowers() const
{
	std::vector<RawTower> allTowers;
	
	TowerIndices indices(0); // Create a set of dummy indices to reuse
	towerID_t& beltIndex = indices.beltIndex;
	towerID_t& phiIndex = indices.phiIndex;	
		
	for(beltIndex = 0; beltIndex < numBelts; ++beltIndex)
	{
		towerID_t const numTowers = DeltaPhi(beltIndex).NumTowers();
		double const fA = surfaceFraction[beltIndex];
		
		for(phiIndex = 0; phiIndex < numTowers; ++phiIndex)
		{
			allTowers.emplace_back(GetTowerEdges(indices), fA);
			
			// Now emplace the corresponding tower in the back detector
			allTowers.emplace_back(allTowers.back());
			allTowers.back().FlipZ();
		}
	}
	
	return allTowers;
}

std::vector<ArrogantDetector::RawTower> ArrogantDetector::AllVisible_InTowers() const
{
	cal_t foreCal_tmp, backCal_tmp;
	
	std::vector<RawTower> visible;
	
	{
		std::vector<vec3_t> particles = tracks;
		particles.insert(particles.end(), towers.begin(), towers.end()); // Is MissingE in here?
			
		for(auto const& p3 : particles)
		{
			double const absTanTheta = ArrogantDetector::AbsTanTheta(p3);
			
			if(absTanTheta < tanThetaMax_cal)
			{
				(IsForward(p3) ? foreCal_tmp : backCal_tmp)
					[GetIndices_AbsTanTheta_Phi(absTanTheta, p3.T().Phi())] += p3.Mag();
			}
		}
	}
	
	for(auto itTower = foreCal_tmp.begin(); itTower not_eq foreCal_tmp.end(); ++itTower)
	{
		TowerID const& towerID = itTower->first;
		double const& energy = itTower->second;
	
		// Correct the energy to make the LEGO plot integratable
		double const fractionalSolidAngle = 1.;
			//~ surfaceFraction(GetIndices_EtaPhi(towerID).first);
		
		visible.push_back(RawTower{GetTowerEdges(towerID), energy / fractionalSolidAngle});
	}
	
	for(auto itTower = backCal_tmp.begin(); itTower not_eq backCal_tmp.end(); ++itTower)
	{
		TowerID const& towerID = itTower->first;
		double const& energy = itTower->second;
		
		// Correct the energy to make the LEGO plot integratables
		double const fractionalSolidAngle = 1.;
			//~ surfaceFraction(GetIndices_EtaPhi(towerID).first);
			
		visible.push_back(RawTower{GetTowerEdges(towerID), energy / fractionalSolidAngle});
		visible.back().FlipZ();
	}
	
	return visible;
}

////////////////////////////////////////////////////////////////////////

ArrogantDetector_Hadron::ArrogantDetector_Hadron(QSettings const& parsedINI, 
	std::string const& detectorName):
ArrogantDetector(parsedINI, detectorName),
deltaPhi(settings.squareWidth, settings.evenTowers)
{
	// We want our towers as square as possible,
	// so we require that the span in theta (deltaTheta) equals the 
	// span in phi (cos(theta_0)*deltaPhi), where theta_0 is 
	// the central theta for each band:
	// 	deltaTheta = deltaPhi * std::cos(theta_0)
	// deltaPhi is fixed for every band.
	// We can solve this transcendental equation recursively, 
	// using the fixed-point iterative method (x = g(x)). 
	// This recursion will converge provided that g'(x0) < 1 in the 
	// vicinity of the fixed point x0.
	// While we do not explicitly check that this condition is satisfied, 
	// empirical testing reveals no problems with reasonable values of squareWidth.
	// We do not need an exact solution, so we define a convergence threshold.
	static constexpr double threshold = 1e-5;

	// Start at the equator
	double eta_lower = 0.;
	double theta_lower = 0.;
	
	// Being near the equator, theta-phi space is rather Euclidean, 
	// so an educated guess for deltaTheta is that it's equal to deltaPhi.
	double deltaTheta = settings.squareWidth;
	
	// Keep defining calorimeter bands till we reach the end of coverage
	while(eta_lower <= settings.etaMax_cal)
	{
		// Each iteration starts with a valid lower edge; store it.
		beltEdges_eta.push_back(eta_lower);
		
																		GCC_IGNORE_PUSH(-Wfloat-equal)
		if(beltEdges_eta.back() == settings.etaMax_cal)
			break;
																		GCC_IGNORE_POP
		
		{
			double deltaTheta_last;
			size_t iterations = 0; // Safety valve for divergence.
			do
			{
				// DeltaPhi is constant; calling DeltaPhi(0) is the easiest way to get it
				deltaTheta_last = deltaTheta;
				deltaTheta = DeltaPhi(0) * std::cos(theta_lower + deltaTheta);
				
				assert(++iterations < 100);
				assert(deltaTheta > 0.);
			}
			// Use the scale-free relative difference to gauge convergence, 
			// in case the two deltaTheta are on vastly different scales
			while(std::fabs(deltaTheta - deltaTheta_last) > threshold * (deltaTheta + deltaTheta_last));
		}
		theta_lower += deltaTheta;
		eta_lower = Theta_to_Eta(theta_lower);
	}
	
	// We have told ArrogantDetector where the equatorial bands are; 
	// it will handle the rest.
	Init_inDerivedCTOR();
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

ArrogantDetector_EqualArea::ArrogantDetector_EqualArea(QSettings const& parsedINI, 
	std::string const& detectorName):
ArrogantDetector(parsedINI, detectorName)
{
	// We will use the same fixed point iterative scheme as ArrogantDetector_Hadron; 
	// (refer to those comments for a primer on the method).
	// This time, however, we want our towers to have approximately constant solid angle
	// 	Omega = \int cos(th) = dPhi * (sin(th2) - sin(th1))
	//          = 2*sin((th2 -th1)/2)*cos((th1 + th2)/2)*dPhi
	// So for the central band (th1 = 0, th2 = dTheta_0)
	//    Omega0 = 2*sin(dTheta_0/2)*cos(dTheta_0/2)*dPhi_0
	// And for all other bands (since all towers share same values)
	//    Omega = 2*sin(dTheta/2)*cos(theta_central)*dPhi	
	static constexpr double threshold = 1e-5;
	
	// We will perform the optimization in two steps;
	//   1. optimize the central band to have square towers
	//   2. optimize the remaining bands to have Omega = Omega0
	
	// Step 1. Map out the first band; start by emplacing its known lower edge and deltaPhi
	beltEdges_eta.push_back(0.);
	
	// deltaPhi has already been determined for the first belt
	EmplaceDeltaPhi(settings.squareWidth);
	
	// We will use the same educated guess for deltaTheta throughout
	double const deltaTheta0 = DeltaPhi(0);
	double deltaTheta = deltaTheta0;
	
	// deltaPhi is fixed; adjust deltaTheta to make the central towers as square as possible
	{
		double deltaTheta_last;
		size_t iterations = 0;
				
		// The same recursion as ArrogantDetector_Hadron
		do
		{
			deltaTheta_last = deltaTheta;
			deltaTheta = DeltaPhi(0) * std::cos(0.5 * deltaTheta);
			
			assert(++iterations < 100);
			assert(deltaTheta > 0.);			
		}
		while(std::fabs(deltaTheta - deltaTheta_last) > threshold * (deltaTheta + deltaTheta_last));
	}	
	
	// Program in the optimized deltaTheta
	double theta_lower = deltaTheta;
	double eta_lower = Theta_to_Eta(theta_lower);
	
	// Calculate the solid angle (dropping the factor of 2 here and everywhere else)
	double const Omega0 = std::sin(0.5*deltaTheta)*std::cos(0.5*deltaTheta)*DeltaPhi(0);
		
	while(eta_lower <= settings.etaMax_cal)
	{
		beltEdges_eta.push_back(eta_lower); // Emplace valid lower edge
		
																		GCC_IGNORE_PUSH(-Wfloat-equal)
		if(beltEdges_eta.back() == settings.etaMax_cal)
			break;
																		GCC_IGNORE_POP
		
		// Testing reveals that the actual deltaTheta always very close to deltaTheta0
		deltaTheta = std::min(deltaTheta0, M_PI_2 - theta_lower);
		
		// Fix deltaPhi based on the educated guess, 
		// (circular continuity gives deltaPhi less freedom; 
		// I fear the situation where we try to co-optimize deltaPhi and deltaTheta, 
		// and deltaPhi begins an infinite oscillate across a half-integer boundary for 
		// the number of phi bins).
		double deltaPhi = Omega0 / (std::sin(0.5*deltaTheta)
			* std::cos(theta_lower + 0.5*deltaTheta));
		deltaPhi = EmplaceDeltaPhi(deltaPhi);
		
		// We now fix deltaPhi and alter deltaTheta to make towers have equal angle
		{
			double deltaTheta_last;
			size_t iterations = 0;
			
			// Now we want our towers to have the same Omega and Omega0
			do
			{
				deltaTheta_last = deltaTheta;
				deltaTheta = 2.*std::asin(Omega0 / (deltaPhi * std::cos(theta_lower + 0.5*deltaTheta)));
				
				assert(++iterations < 1000);
			}
			while(std::fabs(deltaTheta - deltaTheta_last) > threshold * (deltaTheta + deltaTheta_last));
		}
		
		// When we hit the pole, deltaTheta may need to get negative to satisfy omega requirements
		if(deltaTheta > 0.)
		{
			theta_lower += deltaTheta;
			eta_lower = Theta_to_Eta(theta_lower);
		}
		else break;
	}
	
	Init_inDerivedCTOR();
}
		
////////////////////////////////////////////////////////////////////////

ArrogantDetector::PhiSpec const& ArrogantDetector_EqualArea::EmplaceDeltaPhi(double const deltaPhi_target)
{
	belt_deltaPhi.emplace_back(deltaPhi_target, settings.evenTowers);
	return belt_deltaPhi.back();
}

////////////////////////////////////////////////////////////////////////		

ArrogantDetector::PhiSpec const& ArrogantDetector_EqualArea::DeltaPhi(towerID_t const beltIndex) const
{
	return belt_deltaPhi.at(beltIndex);
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////


//~ double ArrogantDetector_Lepton::AbsEquatorialAngle(vec3_t const& particle) const
//~ {
	//~ return std::atan2(std::fabs(particle.x3), particle.T().Mag());
//~ }

////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

//~ double ArrogantDetector_Hadron::AbsEquatorialAngle(vec3_t const& particle) const
//~ {
	//~ return std::fabs(particle.Eta());
//~ }

//~ ////////////////////////////////////////////////////////////////////////

//~ // We have p> = {pT cos(phi), pT sin(phi), pT sinh(eta)} = pT {cos(theta), sin(theta)}
//~ // So theta = atan(sinh(eta)) (also known as the Gudermannian)
//~ double ArrogantDetector_Hadron::ToAbsTheta(double const absEta) const
//~ {
	//~ return std::atan(std::sinh(absEta));
//~ }

////////////////////////////////////////////////////////////////////////



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

////////////////////////////////////////////////////////////////////////

//~ double ArrogantDetector::EquatorialFromAbsEta(double const absEta) const
//~ {
	//~ // To translate \p absEta to the equatorial angle \a t using only AbsEquatorialAngle(),
	//~ // we create a dummy 3-vector. This hack is slow but acceptable since 
	//~ // it is only used during initialization in Init_inDerivedCTOR().
	
	//~ return AbsEquatorialAngle(vec3_t(1., std::fabs(absEta), 0., kdp::Vec3from::LengthEtaPhi));
//~ }

////////////////////////////////////////////////////////////////////////

//~ double ArrogantDetector::EtaFromAbsEquatorial(double const absEquatorial) const
//~ {
	//~ // To translate absEquatorial to pseudorapidity, we create a dummy 3-vector
	//~ // This hack is slow but acceptable since it is only used during initialization	
	//~ double const absTheta = ToAbsTheta(absEquatorial);
	
	//~ return vec3_t(std::cos(absTheta), 0., std::sin(absTheta)).Eta();
//~ }

////////////////////////////////////////////////////////////////////////

//~ double ArrogantDetector::EtaMaxToEquatorialMax(double const etaMax) const
//~ {
	//~ // Convert eta to equatorial angle
	//~ double equatorialMax = kdp::RoundToNearestPitch(EquatorialFromAbsEta(etaMax), settings.squareWidth);
	
	//~ // Make sure there is a beam hole.
	//~ if(equatorialMax >= equatorialMax_geometric)
		//~ equatorialMax -= settings.squareWidth;
	//~ assert(equatorialMax < equatorialMax_geometric);
		
	//~ return equatorialMax;
//~ }

////////////////////////////////////////////////////////////////////////

//~ std::array<double, 4> ArrogantDetector::GetTowerEdges(towerID_t const towerID) const
//~ {
	//~ auto const indices = GetIndices_EquatorialPhi(towerID);
	
	//~ double const absTheta_left = ToAbsTheta(double(indices.first) * settings.squareWidth);
	//~ double const absTheta_right = ToAbsTheta(double(indices.first + 1) * settings.squareWidth);
	//~ assert(absTheta_left >= 0.);
	//~ assert(absTheta_right > absTheta_left);
	//~ assert(absTheta_right < M_PI); // ensuring beam hole
	
	//~ double const deltaPhi = PhiWidth(indices.first);
	//~ double const phi_left = -M_PI + deltaPhi * double(indices.second);
	//~ assert(phi_left >= -M_PI);
	//~ assert(phi_left < M_PI);
	
	//~ return {absTheta_left, absTheta_right, phi_left, phi_left + deltaPhi};
//~ }

////////////////////////////////////////////////////////////////////////

//~ std::vector<ArrogantDetector::vec3_t> ArrogantDetector::GetTowerArea() const
//~ {
	//~ std::vector<vec3_t> towerVec;
	
	//~ /* REMEMBER: we are using a different equatorial angle: th = 0 --> equator, so cos(theta) is the 
	 //~ * dOmega = \int cos(theta) dTheta dPhi = deltaPhi * (sin(t2) - sin(t1))
	 //~ * 	= 2*sin((th2 - th1)/2)*cos((th1 + th2)/2)*dPhi
	 //~ * 2 dPhi / (4 pi) = dPhi / (2 pi) = (2 pi)/numPhiBins / (2 pi) = 1./numPhiBins    so ....
	 //~ * dOmega / (4 pi) = sin((th2 - th1)/2)*cos((th1 + th2)/2) / numPhiBins
	//~ */	
	//~ towerID_t const equatorialIndex_end = NumTowerBands();
		
	//~ for(towerID_t equatorialIndex = 0; equatorialIndex < equatorialIndex_end; ++equatorialIndex)
	//~ {
		//~ towerID_t const numPhiBins = towerID_t(std::round((2.*M_PI) / PhiWidth(equatorialIndex)));
				
		//~ // NO shortcuts; theta and equatorialAngle do not neccesarily map linearly
		//~ double const theta1 = ToAbsTheta(double(equatorialIndex) * settings.squareWidth);
		//~ double const theta2 = ToAbsTheta(double(equatorialIndex + 1) * settings.squareWidth);
		
		//~ // We assume that deltaPhi is the same over an entire band, true up to O(epsilon)
		//~ double const dOmegaNorm = 
			//~ std::sin(0.5*(theta2 - theta1))*std::cos(0.5*(theta1 + theta2)) / double(numPhiBins);
		
		//~ // This is an inefficient way to get the tower center; however, it is the most readable.
		//~ // And we don't expect this function to get called very often
		//~ for(towerID_t phiIndex = 0; phiIndex < numPhiBins; ++phiIndex)
		//~ {
			//~ towerID_t const towerIndex = equatorialIndex * settings.numPhiBins_centralBand + phiIndex;
			//~ towerVec.push_back(GetTowerCenter(towerIndex) * dOmegaNorm);
			//~ towerVec.push_back(towerVec.back());
			//~ towerVec.back().x3 = -towerVec.back().x3; // emplace the backward detector
		//~ }
	//~ }
	
	//~ return towerVec;
//~ }
