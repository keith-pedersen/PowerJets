// Copyright (C) 2018 Keith Pedersen (Keith.David.Pedersen@gmail.com)
// Adapted from PYTHIA main11.cc and main19.cc (licenced under the GNU GPL version 2)

#include "LHE_Pythia_PowerJets.hpp"
//~ #include "SpectralPower.hpp"
#include "kdp/kdpTools.hpp"
#include "kdp/kdpStdVectorMath.hpp"
#include <fstream>

////////////////////////////////////////////////////////////////////////
// LHE_Pythia_PowerJets::Settings
////////////////////////////////////////////////////////////////////////

LHE_Pythia_PowerJets::Settings::Settings(QSettings const& parsedINI)
{
	Main__detectorName.Read(parsedINI);
	
	Pythia__lhePath.Read(parsedINI);
	Pythia__skipEvents.Read(parsedINI);
	Pythia__maxEvents.Read(parsedINI);
	Pythia__abortsAllowed.Read(parsedINI);
	Pythia__confPath.Read(parsedINI);
	
	PowerJets__fR_track.Read(parsedINI);
	PowerJets__u_track.Read(parsedINI);
	
	FastJet__algo.Read(parsedINI);
	FastJet__R.Read(parsedINI);
	
	Pileup__mu.Read(parsedINI);
	Pileup__confPath.Read(parsedINI);	
}

////////////////////////////////////////////////////////////////////////
// LHE_Pythia_PowerJets
////////////////////////////////////////////////////////////////////////

LHE_Pythia_PowerJets::LHE_Pythia_PowerJets(std::string const& INI_filePath):
	LHE_Pythia_PowerJets(QSettings(INI_filePath.c_str(), QSettings::IniFormat)) {}
	
////////////////////////////////////////////////////////////////////////
	
LHE_Pythia_PowerJets::LHE_Pythia_PowerJets(QSettings const& parsedINI):
	// Because of chained defalt arguments, to send a false for printBanner, 
	// we have to send the default value for xmlDir.
	pythia("../xmldoc", false),
	pileup(nullptr), // initializd in main body
	detector(nullptr), // initializd in main body
	clusterAlg(), // initializd in main body
	//~ gen(false), // delay seed
	nextCount(0), abortCount(0),
	status(Status::UNINIT),
	settings(parsedINI)
	//~ trackShape(nullptr), towerShape(nullptr)
{
	////////////////////
	// Initialize Pythia
	{
		if(not std::ifstream(settings.Pythia__lhePath))
		{
			throw std::runtime_error("LHE_Pythia_PowerJets: LHE file <" + 
				settings.Pythia__lhePath.value + "> not found.");
		}

		// If maxEvents == size_t(-1), (maxEvents + skipEvents) <= maxEvents
		// We must ensure that iEvent_end >= skipEvents
		nextCount_max = std::max(settings.Pythia__skipEvents + settings.Pythia__maxEvents, 
			settings.Pythia__maxEvents.value);
		
		// Hard-code LHE analysis
		pythia.readString("Beams:frameType = 4");
		pythia.readString(("Beams:LHEF = " + settings.Pythia__lhePath.value).c_str());
		
		// Pythia seems to choke on LHE e+e- from MadGraph.
		// This flag matches the input energies based upon the output
		pythia.readString("LesHouches:matchInOut = off"); //https://bugs.launchpad.net/mg5amcnlo/+bug/1622747, Stefan Prestel's recommendation
		//pythia.readString("LesHouches:setLeptonMass = 0"); // Take lepton masses from LHE file, doesn't help

		// Read all other flags from a pythia card (potentially overwriting the default values here)
		pythia.readFile(settings.Pythia__confPath.value.c_str());
		pythia.init(); // Start her up

		// If we need to skip events, do it now
	
		// The purpose of skipEvent is to quickly get to (iEvent = skipEvent). 
		// However, for this to be the *exact* same event, 
		// Pythia's PRNG needs to be called the correct number of times.
		// Thus, we actually need to call pythia.next() for the skipped events.				
		while((nextCount < settings.Pythia__skipEvents) and 
			(Next_internal(false) == Status::OK));
	}
	
	//////////////////////////
	// Initialize the detector	
	detector = ArrogantDetector::NewDetector(parsedINI, settings.Main__detectorName);
	
	/////////////////////
	// Initialize fastjet
	{
		static std::map<std::string, fastjet::JetAlgorithm> const algoMap = 
			{
				{"kt", fastjet::kt_algorithm},
				{"CA", fastjet::cambridge_algorithm},
				{"anti-kt", fastjet::antikt_algorithm}};
				
		auto const itAlgo = algoMap.find(settings.FastJet__algo);
		if(itAlgo == algoMap.end())
			throw std::logic_error(std::string("LHE_Pythia_PowerJets: 'fastjet/algo' not recognized. ") + 
				"Only 'kt', 'CA', 'anti-kt' are supported");
				
		clusterAlg = fastjet::JetDefinition(itAlgo->second, settings.FastJet__R);
		
		// Redirect fastjet banner to /dev/null
		fastjet::ClusterSequence::set_fastjet_banner_stream(&fastjet_banner_dummy);
	}
	
	////////////////////
	// Initialize pileup
	{
		if(settings.Pileup__mu > 0.)
		{
			pileup = new Pythia8::Pythia("../xmldoc", false);
			
			if(not std::ifstream(settings.Pileup__confPath).is_open())
				throw std::runtime_error("LHE_Pythia_PowerJets: Pythia conf for pileup not found: <"
					 + settings.Pileup__confPath.value + ">");
					 
			pileup->readFile(settings.Pileup__confPath.value.c_str());
			pileup->init();
		}
	}
	
	// Old, isotropic pileup
	////////////////////
	// Initialize pileup
	//~ pileup_meanF = parsedINI.value("pileup/meanF", 1e-2).toDouble(); // Negative value means no pileup
	//~ pileup_noise2signal = parsedINI.value("pileup/noise2signal", -1.).toDouble(); // Negative value means no pileup
	
	//~ {
		//~ std::string puScheme = parsedINI.value("pileup/balancingScheme", "shim").toString().toStdString();
		
		//~ if(puScheme == "back2back")
			//~ puBalancingScheme = PileupBalancingScheme::back2back;
		//~ else if(puScheme == "shim")
			//~ puBalancingScheme = PileupBalancingScheme::shim;
		//~ else
			//~ throw std::runtime_error("LHE_Pythia_PowerJets: pileup balancing scheme \"" + puScheme + "\" not recognized");
	//~ }
}

////////////////////////////////////////////////////////////////////////

void LHE_Pythia_PowerJets::Clear()
{
	detector->Clear();
	
	detected.clear();	
	Hl_Obs.clear();
	//~ detectorFilter.clear();
		
	fast_jets.clear();	
}

////////////////////////////////////////////////////////////////////////

LHE_Pythia_PowerJets::Status LHE_Pythia_PowerJets::DoWork()
{
	Clear(); // Clear all caches, ESPECIALLY if there's a problem
	
	if(status == Status::OK)
	{
		// Fill the hard scatter
		detector->PartialFill(pythia, false); // isPileup = false
		
		if(pileup)
		{
			size_t const nPileup = size_t(std::ceil(settings.Pileup__mu));
			
			for(size_t k = 0; k < nPileup; ++k) // Fill pileup
			{
				pileup->next();
				detector->PartialFill(*pileup, true); // isPileup = true
			}
		}
		detector->Finalize();
		
		detected = detector->Tracks();
		detected.insert(detected.end(), 
			detector->Towers().cbegin(), detector->Towers().cend());
		
		// Get the observation and make tracks extensive based upon the sample's angular resolution
		tracksTowers = detector->GetObservation().
			MakeExtensive(settings.PowerJets__fR_track, settings.PowerJets__u_track);
	}
	
	return status;
}

////////////////////////////////////////////////////////////////////////

LHE_Pythia_PowerJets::Status LHE_Pythia_PowerJets::Next_internal(bool doWork)
{
	if(status == Status::UNINIT)
		status = Status::OK;
		
	// Returning immediately upon bad status preserves first status change from 
	// other control logic (e.g., END_OF_FILE takes priority over EVENT_MAX)	
	if(status == Status::OK)
	{
		// Loop while we have more events to generate AND Pythia is
		// aborting the event. Every iteration calls pythia.next() one more time
		while((nextCount < nextCount_max) and 
			not (++nextCount, pythia.next())) // comma operator; do first command, return second
			// We use comma op this to tie ++iEvent to every call to next()
		{
			// If failure because reached end of file, return
			if(pythia.info.atEndOfFile())
				return (status = Status::END_OF_FILE);
				
			// First few bad events write off as "acceptable" errors
			if(++abortCount < settings.Pythia__abortsAllowed)
				continue;
			else // otherwise return failure
				return (status = Status::ABORT_MAX);
		}
		
		// Check for too many events
		if(nextCount >= nextCount_max)
			return (status = Status::EVENT_MAX);
	}
	
	if(doWork)
		return DoWork(); // Return status after doing work
	else
		return status;
}

////////////////////////////////////////////////////////////////////////

void LHE_Pythia_PowerJets::ClusterJets() const
{
	// Fill the particle into a format useful by fastjet
	std::vector<fastjet::PseudoJet> protojets;
	{
		for(vec3_t const& particle : detected)
		{
			// energy goes last
			protojets.emplace_back(particle.x1, particle.x2, particle.x3, particle.Mag());
		}
	}
	
	// run the clustering, extract the jets
	fastjet::ClusterSequence cs(protojets, clusterAlg);
	std::vector<fastjet::PseudoJet> const jets = fastjet::sorted_by_E(cs.inclusive_jets());
	
	// We use insert to force an element-wise call to Jet(fastjet::PseudoJet const&)
	fast_jets.insert(fast_jets.end(), jets.cbegin(), jets.cend());
}

////////////////////////////////////////////////////////////////////////

LHE_Pythia_PowerJets::~LHE_Pythia_PowerJets() 
{
	delete detector;
	delete pileup;
}

////////////////////////////////////////////////////////////////////////

std::vector<Jet> const& LHE_Pythia_PowerJets::Get_FastJets() const
{
	if(fast_jets.empty()) // cache the first call
		ClusterJets();
	
	return fast_jets;
}

////////////////////////////////////////////////////////////////////////

std::vector<LHE_Pythia_PowerJets::real_t> const&
LHE_Pythia_PowerJets::Get_Hl_Obs(size_t const lMax) const
{
	if(Hl_Obs.size() < lMax) 
		Hl_Obs = Hl_computer.Hl_Obs(lMax, tracksTowers); // cache the first call
	return Hl_Obs;
}

////////////////////////////////////////////////////////////////////////

std::vector<LHE_Pythia_PowerJets::real_t>
LHE_Pythia_PowerJets::Get_Hl_Jet(size_t const lMax, std::vector<ShapedJet> const& jets) const
{
	return Hl_computer.Hl_Jet(lMax, jets, std::vector<real_t>()); //Get_DetectorFilter(lMax));
}

////////////////////////////////////////////////////////////////////////

std::vector<LHE_Pythia_PowerJets::real_t>
LHE_Pythia_PowerJets::Get_Hl_Hybrid(size_t const lMax, std::vector<ShapedJet> const& jets) const
{
	return Hl_computer.Hl_Hybrid(lMax, jets, 
		std::vector<real_t>(), //Get_DetectorFilter(lMax), 
		tracksTowers, Get_Hl_Obs(lMax));
}

////////////////////////////////////////////////////////////////////////

void LHE_Pythia_PowerJets::WriteAllVisibleAsTowers(std::string const& filePath) const
{
	std::ofstream file(filePath, std::ios::trunc);
	
	// Write to a *.dat with space/tab separated for easy Mathematica import
	if(file.is_open())
	{
		char buffer[1024];
		
		auto const allAsTowers = detector->AllVisible_InTowers();
		
		for(auto const& tower : allAsTowers)
		{
			sprintf(buffer, "% .6e % .6e % .6e % .6e % .6e\n", 
				tower.eta_lower, tower.eta_upper, tower.phi_lower, tower.phi_upper,
				tower.energy);
			file << buffer;
		}		
	}
}

////////////////////////////////////////////////////////////////////////

//~ LHE_Pythia_PowerJets::vec4_t LHE_Pythia_PowerJets::IsoVec3_Exponential
	//~ (pqRand::engine& gen, real_t const meanE) 
//~ {
	//~ // Easiest way to draw an isotropic vec3 is rejection sampling
	
	//~ vec4_t iso(false); // Don't initialize (false)
	//~ real_t r2;
	
	//~ do
	//~ {
		//~ // U_S has enough precision for this application
		//~ iso.x1 = gen.U_even();
		//~ iso.x2 = gen.U_even();
		//~ iso.x3 = gen.U_even();
		
		//~ r2 = iso.p().Mag2();
	//~ }
	//~ while(r2 > real_t(1));
	
	//~ // Now we have a vector whose direction is guarenteed to be isotropic, 
	//~ // but whose length is not only wrong but seems totally useless;
	//~ // we will downscale the vector by its current length when we resize it to
	//~ // a length randomly drawn from the exponential distribution.
	//~ // However, its current length is not entirely useless. 
	//~ // It is in fact a random length drawn from some predictable distribution.
	//~ // So what if we can use it as the input for the quantile function that
	//~ // maps out the exponential distribution. This way we can save 
	//~ // drawing another random number.
	//~ // (NOTE: We will have drawn three random numbers, very much like 
	//~ //  if we had drawn theta, phi, and a random length.
	//~ //  So we can see that we still have to draw the same # of d.o.f.).
	
	//~ // First we must figure out the distribution of r^2 from volume element
	//~ // 	dV / V = (4 pi r**2 dr ) / (4/3 pi R**3)   (R = 1)
	//~ // 	dV / (V dr) = 3 r**2     r2 = r**2    dr2 = 2 r dr
	//~ // 	dV / (V dr2) = 3/2 r2**(1/2)
	//~ // Now we solve for the CDF of r2 (CFD(r2) = \int_{r2p=0}^{r2p=r2}r2p**(3/2)
	//~ // 	u = CDF(r2) = (r2)**(3/2)
	//~ // Now we can plug this uniformly distributed u
	//~ // into the quantile function for an exponential distribution
	//~ //    Q1(u) = -log(u)/lambda    Q2(u) = -log1p(u)/lambda
	//~ // If we want to implement a quantile flip-flop, we simply
	//~ // halve the output of the CDF
	
	//~ {
		//~ real_t const hu = real_t(0.5) * std::pow(r2, real_t(1.5));
		//~ real_t const energy = meanE * (gen.RandBool() ? -std::log(hu): -std::log1p(-hu));
	
		//~ // Now scale the vector to the new length (dividing by current)
		//~ iso *= (energy / std::sqrt(r2));
		//~ iso.x0 = energy;
	//~ }
		
	//~ // Now move to the other 7 octants
	//~ gen.ApplyRandomSign(iso.x1);
	//~ gen.ApplyRandomSign(iso.x2);
	//~ gen.ApplyRandomSign(iso.x3);
	
	//~ return iso;
//~ }

//~ std::vector<LHE_Pythia_PowerJets::real_t> const&
//~ LHE_Pythia_PowerJets::Get_Hl_FinalState(size_t const lMax)
//~ {
	//~ if(Hl_FinalState.size() <= lMax)
	//~ {
		//~ auto trackShape = std::shared_ptr<ShapeFunction>(new h_PseudoNormal(
			//~ f_trackR*observation.NaiveObservation().AngularResolution(), 
			//~ u_trackR));
		
		//~ Hl_FinalState = Hl_computer.Hl_Obs(lMax,
			//~ ShapedParticleContainer(VecPhatF(detector->FinalState()), trackShape));
	//~ }
	//~ return Hl_FinalState;
//~ }

//~ std::vector<LHE_Pythia_PowerJets::real_t>
//~ LHE_Pythia_PowerJets::Get_Hl_Obs_slow(size_t const lMax)
//~ {
	//~ return SpectralPower::Hl_Obs(lMax, tracks, *trackShape, towers, *towerShape);
//~ }

//~ std::vector<LHE_Pythia_PowerJets::real_t>
//~ LHE_Pythia_PowerJets::Get_Hl_Jet_slow(size_t const lMax, std::vector<ShapedJet> const& jets)
//~ {
	//~ return SpectralPower::Hl_Jet(lMax, jets, Get_DetectorFilter(lMax));
//~ }

//~ std::vector<LHE_Pythia_PowerJets::real_t>
//~ LHE_Pythia_PowerJets::Get_Hl_Hybrid_slow(size_t const lMax, std::vector<ShapedJet> const& jets)
//~ {
	//~ return SpectralPower::Hl_Hybrid(lMax, jets, Get_DetectorFilter(lMax),
		//~ tracks, *trackShape, towers, *towerShape, Get_Hl_Obs(lMax));
//~ }

//~ std::vector<LHE_Pythia_PowerJets::real_t> const& LHE_Pythia_PowerJets::Get_DetectorFilter(size_t const lMax)
//~ {
	//~ if(detectorFilter.size() < lMax)
	//~ {
		//~ std::vector<real_t> hl_track = trackShape->hl_Vec(lMax);
		//~ std::vector<real_t> hl_tower = towerShape->hl_Vec(lMax);
		
		//~ hl_track *= chargeFraction;
		//~ hl_tower *= real_t(1) - chargeFraction;
		
		//~ detectorFilter = hl_track + hl_tower;
	//~ }
	
	//~ return detectorFilter;
//~ }

//~ void LHE_Pythia_PowerJets::MakePileup()
//~ {
	//~ pileup.clear();
	
	//~ if(pileup_noise2signal > real_t(0))
	//~ {
		//~ real_t const pu_totalE_target = pileup_noise2signal * pythia.event.scale();
		//~ real_t const pu_meanE = pileup_meanF * pythia.event.scale();
		
		//~ assert(pu_meanE > real_t(0));
		
		//~ real_t pu_TotalE = real_t(0);
		//~ real_t pu_maxE = real_t(-1);
						
		//~ // Ensuring that pileup sums to zero is not trivial.
		//~ // Adding up isotropic 3-vectors creates a 3D random walk, 
		//~ // so while we expect (sum/n) to converge to zero, (sum) itself will diverge.
		//~ // I have found 2 methods which keep sum balanced:
		//~ //    1. Quick and dirty; draw a vector, add its opposite.
		//~ // 	2. Slightly less dirty; draw 2 vectors, add the opposite of their sum.
		//~ //		3. Monitor sum and whenever sum.Mag() gets too large,
		//~ //			"shim" it by adding a unit vector opposite of sum. 
		//~ //       Leave space for 2 unit vectors to neutralize the final sum.
		//~ // This latter was designed for isotropic unit vectors, so we will not use it.
		//~ // A quick study in Mathematica (ExponentialPileupBalancing) shows that
		//~ // method #2 does not egregiously alter the exponential distribution,
		//~ // only shifting up it's mean by about 10%
		//~ // (and slightly diminishing the probability of zero-energy particles).
		
		//~ switch(puBalancingScheme)
		//~ {
			//~ case PileupBalancingScheme::back2back:
				//~ while(pu_TotalE < pu_totalE_target)
				//~ {
					//~ pileup.push_back(IsoVec3_Exponential(gen, pu_meanE));
					//~ pileup.emplace_back(pileup.back().x0, -pileup.back().p(), kdp::Vec4from2::Energy);
					
					//~ pu_TotalE += 2. * pileup.back().x0;
					//~ pu_maxE = std::max(pu_maxE, pileup.back().x0);
					
					//~ // if(pileup.size() > (size_t(1) << 10))
						//~ // throw std::runtime_error("problem");
				//~ }
			//~ break;
			
			//~ case PileupBalancingScheme::shim:
				//~ while(pu_TotalE < pu_totalE_target)
				//~ {
					//~ vec3_t sum;
					
					//~ for(size_t i = 0; i < 2; ++i)
					//~ {
						//~ pileup.push_back(IsoVec3_Exponential(gen, pu_meanE));
						//~ sum += pileup.back().p();
						
						//~ pu_TotalE += pileup.back().x0;
						//~ pu_maxE = std::max(pu_maxE, pileup.back().x0);
					//~ }
					
					//~ pileup.emplace_back(0., -sum, kdp::Vec4from2::Mass);
					//~ pu_TotalE += pileup.back().x0;
					//~ pu_maxE = std::max(pu_maxE, pileup.back().x0);
				//~ }
			//~ break;
		//~ }
			
		//~ // THe lesson below: Added naively, pileup's random walk 
		//~ // creates a large momentum imbalance which cannot be accounted for 
		//~ // by the addition of a single particle opposite the total, 
		//~ // because the energy of that single particle is way too large.
		//~ // The results of the first few events examined:
		//~ // 1-CDF: {1.886e-03, 3.987e-04, 8.038e-05, 1.551e-05, 1.762e-04, 5.858e-03, 9.268e-08}
		//~ //
		//~ // CDF: 1 − e−λx
		//~ // λ = 1 / pu_meanE
		//~ // double const puTotalP = std::accumulate(pileup.begin(), pileup.end(), vec3_t()).Mag();
		//~ // printf("1-CDF: %.3e\n", std::exp(-puTotalP/pu_meanE));
	//~ }
//~ }
