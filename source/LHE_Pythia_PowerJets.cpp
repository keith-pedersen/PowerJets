// Copyright (C) 2018 Keith Pedersen (Keith.David.Pedersen@gmail.com)
/* Adapted from: 
 *  - Pythia's main11.cc and main19.cc (licenced under the GNU GPLv2)
 *  - FastJet's 07-subtraction.cc (also licensed under the GNU GPLv2)
 * Therefore, this file is also licensed under the GNU GPLv2
*/ 

#include "LHE_Pythia_PowerJets.hpp"
//~ #include "SpectralPower.hpp"
#include "kdp/kdpTools.hpp"
#include "kdp/kdpStdVectorMath.hpp"

#include "fastjet/ClusterSequenceArea.hh"

#include <fstream>

////////////////////////////////////////////////////////////////////////
// LHE_Pythia_PowerJets::Settings
////////////////////////////////////////////////////////////////////////

LHE_Pythia_PowerJets::Settings::Settings(QSettings const& parsedINI)
{
	Main__detectorName.Read(parsedINI);
	Main__printBanners.Read(parsedINI);
	
	Pythia__lhePath.Read(parsedINI);
	Pythia__skipEvents.Read(parsedINI);
	Pythia__maxEvents.Read(parsedINI);
	Pythia__abortsAllowed.Read(parsedINI);
	Pythia__confPath.Read(parsedINI);
	
	PowerJets__fR_track.Read(parsedINI);
	PowerJets__u_track.Read(parsedINI);
	
	FastJet__algo.Read(parsedINI);
	FastJet__R.Read(parsedINI);
	FastJet__algo_pileup.Read(parsedINI);
	FastJet__R_pileup.Read(parsedINI);
	FastJet__nHardExclude_pileup.Read(parsedINI);
	
	Pileup__mu.Read(parsedINI);
	Pileup__confPath.Read(parsedINI);
	Pileup__hlPath.Read(parsedINI);
}

////////////////////////////////////////////////////////////////////////
// LHE_Pythia_PowerJets
////////////////////////////////////////////////////////////////////////

LHE_Pythia_PowerJets::LHE_Pythia_PowerJets(std::string const& INI_filePath):
	LHE_Pythia_PowerJets(QSettings(INI_filePath.c_str(), QSettings::IniFormat)) {}
	
////////////////////////////////////////////////////////////////////////
	
LHE_Pythia_PowerJets::LHE_Pythia_PowerJets(QSettings const& parsedINI):
	settings(parsedINI),
	// Because of chained defalt arguments, to send a false for printBanner, 
	// we have to send the default value for xmlDir.
	pythia(Pythia_xmlDoc, settings.Main__printBanners),
	pileup(nullptr), // initializd in main body
	detector(nullptr), // initializd in main body
	//~ gen(false), // delay seed
	nextCount(0), abortCount(0),
	status(Status::UNINIT)	
{
	/////////////////////////////////////////
	// Print banners (Pythia already printed)
	if(settings.Main__printBanners)
	{
		fastjet::ClusterSequenceArea::print_banner();
		std::cout << "\n";
	}
	
	Initialize_Pythia();	
		
	//////////////////////////
	// Initialize the detector	
	detector = ArrogantDetector::NewDetector(parsedINI, settings.Main__detectorName);
	//~ std::cout << detector->GetSettings().etaMax_cal << std::endl;
	
	Initialize_FastJet();
	
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
		
	//~ fast_jets.clear();
	
	pileupEstimator.set_particles(std::vector<fastjet::PseudoJet>());
}

////////////////////////////////////////////////////////////////////////

void LHE_Pythia_PowerJets::Set_PileupMu(double const pileup_mu)
{
	settings.Pileup__mu = pileup_mu;
	Warmup_Pileup(); // In case Pileup__mu was formerly zero
}

////////////////////////////////////////////////////////////////////////

void LHE_Pythia_PowerJets::Initialize_Pythia()
{
	/////////////////////////////////////////////////////////////////////
	// Main Pythia
	/////////////////////////////////////////////////////////////////////
	
	if(not std::ifstream(settings.Pythia__lhePath).is_open())
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
	
	if(not pythia.init()) // Start her up
		throw std::runtime_error("Pythia can't initialize");
	
	// If we need to skip events, do it now
	// The purpose of skipEvent is to quickly get to (iEvent = skipEvent).
	// However, for this to be the *exact* same event, 
	// Pythia's PRNG needs to be called the correct number of times.
	// Thus, we actually need to call pythia.next() for the skipped events.	
	while((nextCount < settings.Pythia__skipEvents) and 
		(Next_internal(false) == Status::OK)); // doWork = false
	
	/////////////////////////////////////////////////////////////////////
	// Pileup Pythia
	/////////////////////////////////////////////////////////////////////
	Warmup_Pileup();
	
	if(settings.Pileup__hlPath.value.size()) // Check for non-empty string (the default)
		pileupShape = ShapeFunction::Make<h_Measured>(settings.Pileup__hlPath);
}

////////////////////////////////////////////////////////////////////////

void LHE_Pythia_PowerJets::Warmup_Pileup()
{
	if(pileup) // We've already created and initialized the pileup Pythia
		return;
	else if(settings.Pileup__mu > 0.)
	{
		// We only create the pythia instance when pileup is activated, 
		// since the warm-up procedure takes time.
		pileup = new Pythia8::Pythia(Pythia_xmlDoc, false); // false = Don't print the Pythia banner
		
		if(not std::ifstream(settings.Pileup__confPath).is_open())
			throw std::runtime_error("LHE_Pythia_PowerJets: Pythia conf for pileup not found: <"
				 + settings.Pileup__confPath.value + ">");
				 
		pileup->readFile(settings.Pileup__confPath.value.c_str());
		printf("Warming up Pythia for soft QCD (pileup); this may take a few seconds ...\n");
		pileup->init();
		printf("-------------------------------------> Pileup warmup done\n");
	}
}

////////////////////////////////////////////////////////////////////////

void LHE_Pythia_PowerJets::Initialize_FastJet()
{
	static std::map<std::string, fastjet::JetAlgorithm> const algoMap = {
		{"kt", fastjet::kt_algorithm},
		{"CA", fastjet::cambridge_algorithm},
		{"anti-kt", fastjet::antikt_algorithm}};
		
	// Redirect fastjet banner to /dev/null, because we have already printed it if requested
	fastjet::ClusterSequenceArea::set_fastjet_banner_stream(&fastjet_banner_dummy);
		
	// Create a local function to emit an exception when the algorithm is not found
	static auto algoError = [](Settings::Param<std::string> const& param)
	{
		throw std::logic_error(std::string("LHE_Pythia_PowerJets: key '") + 
			param.key + "' value '" + param.value + "' not recognized. "
			+ "Only 'kt', 'CA', 'anti-kt' are supported");
	};
	
	// Initialize the main clustering algorithm
	{
		auto const itAlgo = algoMap.find(settings.FastJet__algo);
		if(itAlgo == algoMap.end())
			algoError(settings.FastJet__algo);
			
		clusterAlg = fastjet::JetDefinition(itAlgo->second, settings.FastJet__R);
	}
	
	// Initialize the pileup clustering algorithm
	{
		auto const itAlgo_pileup = algoMap.find(settings.FastJet__algo_pileup);
		if(itAlgo_pileup == algoMap.end())
			algoError(settings.FastJet__algo_pileup);
			
		clusterAlg_pileup = fastjet::JetDefinition(itAlgo_pileup->second, settings.FastJet__R_pileup);
	}
	
	//////////////////////////////////////////////////////////////////////
	// The following comments were lifted from FastJet's 07-subtraction.cc,
	// which I pretty much copied verbatim (with renamed variables)
	// to subtract pileup energy
	
	// Now turn to the estimation of the background (for the full event)
	//
	// There are different ways to do that. In general, this also
	// requires clustering the particles that will be handled internally
	// in FastJet. 
	//
	// The suggested way to proceed is to use a BackgroundEstimator
	// constructed from the following 3 arguments:
	//  - a jet definition used to cluster the particles.
	//    . We strongly recommend using the kt or Cambridge/Aachen
	//      algorithm (a warning will be issued otherwise)
	//    . The choice of the radius is a bit more subtle. R=0.4 has
	//      been chosen to limit the impact of hard jets; in samples of
	//      dominantly sparse events it may cause the UE/pileup to be
	//      underestimated a little, a slightly larger value (0.5 or
	//      0.6) may be better.
	//  - An area definition for which we recommend the use of explicit
	//    ghosts (i.e. active_area_explicit_ghosts)
	//    As mentionned in the area example (06-area.cc), ghosts should
	//    extend sufficiently far in rapidity to cover the jets used in
	//    the computation of the background (see also the comment below)
	//  - A Selector specifying the range over which we will keep the
	//    jets entering the estimation of the background (you should
	//    thus make sure the ghosts extend far enough in rapidity to
	//    cover the range, a warning will be issued otherwise).
	//    In this particular example, the two hardest jets in the event
	//    are removed from the background estimation
	// ----------------------------------------------------------

	// create an area definition for the clustering
	//----------------------------------------------------------
	// ghosts should go up to the acceptance of the detector or
	// (with infinite acceptance) at least 2R beyond the region
	// where you plan to investigate jets.
	double const maxRapditiy_det = detector->GetSettings().etaMax_cal;
	double const maxRapidity_ghost = maxRapditiy_det
		+ 2.*std::max(settings.FastJet__R, settings.FastJet__R_pileup);
	double const maxRapidity_pileup = maxRapditiy_det
		- std::max(settings.FastJet__R, settings.FastJet__R_pileup);
		
	areaDef = fastjet::AreaDefinition(fastjet::active_area, 
		fastjet::GhostedAreaSpec(maxRapidity_ghost));
	
	areaDef_pileup = fastjet::AreaDefinition(fastjet::active_area_explicit_ghosts, 
		fastjet::GhostedAreaSpec(maxRapidity_ghost));
	
	pileupSelector = fastjet::SelectorAbsRapMax(maxRapidity_pileup)
		* (!fastjet::SelectorNHardest((unsigned int)settings.FastJet__nHardExclude_pileup.value));
		
	pileupEstimator = fastjet::JetMedianBackgroundEstimator(pileupSelector, 
		clusterAlg_pileup, areaDef_pileup);
		
	//~ std::cout << pileupEstimator.use_area_4vector() << std::endl;
	
	// To help manipulate the background estimator, we also provide a
	// transformer that allows to apply directly the background
	// subtraction on the jets. This will use the background estimator
	// to compute rho for the jets to be subtracted.
	// ----------------------------------------------------------
	pileupSubtractor = fastjet::Subtractor(&pileupEstimator);

	// since FastJet 3.1.0, rho_m is supported natively in background
	// estimation (both JetMedianBackgroundEstimator and
	// GridMedianBackgroundEstimator).
	//
	// For backward-compatibility reasons its use is by default switched off
	// (as is the enforcement of m>0 for the subtracted jets). The
	// following 2 lines of code switch these on. They are strongly
	// recommended and should become the default in future versions of
	// FastJet.
	pileupSubtractor.set_use_rho_m(true);
	pileupSubtractor.set_safe_mass(true);
}

////////////////////////////////////////////////////////////////////////

LHE_Pythia_PowerJets::Status LHE_Pythia_PowerJets::DoWork(METcorrection const method)
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
		detector->Finalize(method);
		
		detected = detector->Tracks();
		detected.insert(detected.end(), 
			detector->Towers().cbegin(), detector->Towers().cend());
		
		// Get the observation and make tracks extensive based upon the sample's angular resolution
		auto observation = detector->GetObservation();
		assert(kdp::AbsRelError(observation.fTotal(), real_t(1)) < 1e-14);
				
		tracksTowers = observation.MakeExtensive(Hl_computer.AngularResolution(observation),
			settings.PowerJets__fR_track, settings.PowerJets__u_track);
	}
	return status;
}

////////////////////////////////////////////////////////////////////////

LHE_Pythia_PowerJets::Status LHE_Pythia_PowerJets::Next_internal(bool doWork,
	METcorrection const method)
{
	if(status == Status::UNINIT)
		status = Status::OK;
	
	// Returning immediately upon bad status preserves first status change from 
	// other control logic (e.g., END_OF_FILE takes priority over EVENT_MAX)	
	if(status == Status::OK)
	{
		// Check for too many events (do this first so the last/max event returns OK)
		if(nextCount >= nextCount_max)
			return (status = Status::EVENT_MAX);
		
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
	}
	
	if(doWork)
		return DoWork(method); // Return status after doing work
	else
		return status;
}

////////////////////////////////////////////////////////////////////////

std::vector<Jet> LHE_Pythia_PowerJets::Cluster_FastJets(bool const subtractPileup) const
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
	fastjet::ClusterSequenceArea clusterSequence(protojets, clusterAlg, areaDef);
	std::vector<fastjet::PseudoJet> jets = fastjet::sorted_by_E(clusterSequence.inclusive_jets());
	
	if(subtractPileup)
	{
		// Only towers should be used for pileup estimation, since tracks are NOT pileup
		
		std::vector<fastjet::PseudoJet> protojets_cal;
		{
			for(vec3_t const& particle : detector->Towers())
			{
				// energy goes last
				protojets_cal.emplace_back(particle.x1, particle.x2, particle.x3, particle.Mag());
			}
		}
		
		// Finally, once we have an event, we can just tell the background
		// estimator to use that list of particles
		// This could be done directly when declaring the background
		// estimator but the usage below can more easily be accomodated to a
		// loop over a set of events.
		// ----------------------------------------------------------
		pileupEstimator.set_particles(protojets_cal);
		
		jets = fastjet::sorted_by_E(pileupSubtractor(jets));
		
		// Prune out the null (pileup) jets
		jets.resize(std::lower_bound(jets.cbegin(), jets.cend(), 0., 
			[](fastjet::PseudoJet const& jet, double const val){return jet.E() > val;})-jets.cbegin());
	}
	
	// We use insert to force an element-wise call to Jet(fastjet::PseudoJet const&)
	return std::vector<Jet>(jets.cbegin(), jets.cend());
}

double LHE_Pythia_PowerJets::Get_RhoPileup() const
{
	return pileupEstimator.rho();
}

////////////////////////////////////////////////////////////////////////

LHE_Pythia_PowerJets::~LHE_Pythia_PowerJets() 
{
	delete detector;
	delete pileup;
}

////////////////////////////////////////////////////////////////////////

//~ std::vector<Jet> const& LHE_Pythia_PowerJets::Get_FastJets(bool const subtractPileup) const
//~ {
	//~ if(fast_jets.empty()) // cache the first call
		//~ fast_jets = ClusterJets(subtractPileup);
	
	//~ return fast_jets;
//~ }

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
LHE_Pythia_PowerJets::Get_Hl_ME(size_t const lMax) const
{
	return Hl_computer.Hl_Obs(lMax, VecPhatF(detector->ME(), true)); // normalize = true
}

////////////////////////////////////////////////////////////////////////

PowerSpectrum::ShapedParticleContainer LHE_Pythia_PowerJets::JetContainer
	(std::vector<ShapedJet> const& jets, real_t const f_pileup) const
{
	if(f_pileup >= real_t(1))
		throw std::runtime_error("LHE_Pythia_PowerJets::JetContainer: A pileup fraction >= 1 is not feasible");
	if(f_pileup < real_t(0))
		throw std::runtime_error("LHE_Pythia_PowerJets::JetContainer: A negative pileup fraction is not feasible");
		
	ShapedParticleContainer container(jets, false); // normalizeF = false
	
	if(f_pileup > real_t(0))
	{
		if(not(pileupShape))
			throw std::runtime_error("LHE_Pythia_PowerJets::JetContainer: Pileup not initialized.");
		
		// This ensures that the jet ensembled has fTotal() == 1 - f_pileup
		real_t const normalizeFactor = container.fTotal() / (real_t(1) - f_pileup);
		container.NormalizeF_to(normalizeFactor); 
			
		container.emplace_back(PhatF(vec3_t(0, 0, f_pileup)), pileupShape);
	}
	else
		container.NormalizeF();
		
	assert(std::fabs(container.fTotal() - real_t(1)) < 1e-15);
		
	return container;
}
			
////////////////////////////////////////////////////////////////////////

std::vector<LHE_Pythia_PowerJets::real_t>
LHE_Pythia_PowerJets::Get_Hl_Jet(size_t const lMax, std::vector<ShapedJet> const& jets, 
	real_t const f_pileup) const
{
	return Hl_computer.Hl_Jet(lMax, JetContainer(jets, f_pileup),
		std::vector<real_t>()); //Get_DetectorFilter(lMax));
}

////////////////////////////////////////////////////////////////////////

std::vector<LHE_Pythia_PowerJets::real_t>
LHE_Pythia_PowerJets::Get_Hl_Hybrid(size_t const lMax, std::vector<ShapedJet> const& jets, 
	real_t const f_pileup) const
{
	return Hl_computer.Hl_Hybrid(lMax, JetContainer(jets, f_pileup),
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
