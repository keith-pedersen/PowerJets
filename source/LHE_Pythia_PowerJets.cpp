#include "LHE_Pythia_PowerJets.hpp"

void LHE_Pythia_PowerJets::ClearCache()
{
	detected.clear();
	H_det.clear();
	fast_jets.clear();
	ME_vec.clear();
}

LHE_Pythia_PowerJets::LHE_Pythia_PowerJets(std::string const& ini_filePath):
	// Because of chained defalt arguments, to send a false for printBanner, 
	// we have to send the default value for xmlDir.
	pythia("../xmldoc", false),
	parsedINI(ini_filePath.c_str(), QSettings::NativeFormat),
		// Note: Must pass NativeFormat, otherwise it doesn't work.
	detector(ArrogantDetector::NewDetector(parsedINI, "detector")),
		// The detector's parameters are read from the INI folder [detector]
	clusterAlg(), // initializd in main body			
	Hcomputer(parsedINI),
		// HComputer's settings are read from the INI folder [power]
	gen(false), // initialzed in main body
	iEvent_plus1(0),
	lMax(parsedINI.value("power/lMax", 128).toInt()),
	status(Status::UNINIT)
{
	// We could be using a file on disk which uses the default name.
	// We will only use all default values when ini_filePath is not on disk.
	if(not std::ifstream(ini_filePath.c_str()))
		std::cerr << "\nWarning: No configuration file supplied ... everything default values.\n\n";
	
	/////////////////////
	// Initialize fastjet
	{
		static std::map<std::string, fastjet::JetAlgorithm> algoMap = 
			{
				{"kt", fastjet::kt_algorithm},
				{"CA", fastjet::cambridge_algorithm},
				{"anti-kt", fastjet::antikt_algorithm}};
				
		std::string const algo = parsedINI.value("fastjet/algo", "anti-kt").toString().toStdString();
		double const R = parsedINI.value("fastjet/R", 0.4).toDouble();
		
		if(algoMap.find(algo) == algoMap.end())
			throw std::logic_error(std::string("LHE_Pythia_PowerJets: 'fastjet/algo' not recognized. ") + 
				"Only 'kt', 'CA', 'anti-kt' are supported");
				
		clusterAlg = fastjet::JetDefinition(algoMap[algo], R);
		
		// Redirect fastjet banner to /dev/null
		fastjet::ClusterSequence::set_fastjet_banner_stream(&fastjet_banner_dummy);
	}
	
	////////////////////
	// Initialize Pythia
	{
		std::string const lheFile =
			parsedINI.value("main/lheFile", 
				"./unweighted_events.lhe.gz").toString().toStdString();

		if(not std::ifstream(lheFile.c_str()))
		{
			throw std::runtime_error(std::string("LHE_Pythia_PowerJets: LHE file <") + 
				lheFile + "> not found.");
		}

		size_t const skipEvents = size_t(parsedINI.value("main/skipEvents", 0).toInt());
		iEvent_end = skipEvents + size_t(parsedINI.value("main/maxEvents", -1).toInt());
		iEvent_plus1 = 0;
		// -1 means do all, as the size_t will become the largest possible number, 
		// and the Pythia loop will end when the LHE returns end-of-file
		
		// Allow for possibility of a few faulty events.
		abortsAllowed = size_t(parsedINI.value("main/abortsAllowed", 10).toInt());
		abortCount = 0;

		// Hard-code LHE analysis
		pythia.readString("Beams:frameType = 4");
		pythia.readString(("Beams:LHEF = " + lheFile).c_str());
		
		// Pythia seems to choke on LHE e+e- from MadGraph.
		// This flag matches the input energies based upon the output
		pythia.readString("LesHouches:matchInOut = off"); //https://bugs.launchpad.net/mg5amcnlo/+bug/1622747, Stefan Prestel's recommendation
		//pythia.readString("LesHouches:setLeptonMass = 0"); // Take lepton masses from LHE file, doesn't help

		// Read all other flags from a pythia card, changed infrequently (for debugging)
		pythia.readFile(parsedINI.value("main/pythiaConf", "./Pythia.conf").toString().toStdString().c_str());
		
		// Star her up
		pythia.init();

		// If we need to skip events, do it now
	
		// The purpose of skipEvent is to quickly get to (iEvent = skipEvent). 
		// However, for this to be the *exact* same event, 
		// Pythia's PRNG needs to be called the correct number of times.
		// Thus, we actually need to call pythia.next() for the skipped events.				
		while((EventIndex() < skipEvents) and (Next_internal(true) == Status::OK));
	}
	
	////////////////////
	// Initialize pileup
	//~ double const pileup_meanF = parsedINI.value("pileup/meanF", .01).toDouble(); // Negative value means no pileup
	//~ double const pileup_noise2signal = parsedINI.value("pileup/noise2signal", -1.).toDouble(); // Negative value means no pileup
}

LHE_Pythia_PowerJets::Status LHE_Pythia_PowerJets::Next_internal(bool skipAnal)
{
	if(status == Status::UNINIT)
		status = Status::OK;
	
	if(status == Status::OK)
	{
		// iEvent_plus1 enters as iEvent (i.e. it hasn't been incremented yet)
		while((iEvent_plus1 < iEvent_end) and 
			not (++iEvent_plus1, pythia.next())) // comma operator; do first command, return second
			// We use comma op this to tie ++iEvent to every call to next()
		{
			// Loop while Pythia fails
			
			// If failure because reached end of file, then exit event loop.
			if (pythia.info.atEndOfFile())
			{
				status = Status::END_OF_FILE;
				break;
			}
				
			// First few failures write off as "acceptable" errors
			if(++abortCount < abortsAllowed) continue;
			else 
			{
				status = Status::ABORT_MAX;
				break;
			}
		}
		
		// Check for too many events
		if((iEvent_plus1) > iEvent_end)
			status = Status::EVENT_MAX;
	}
	
	if(skipAnal)
		return status;				
	
	ClearCache();
	
	if(status == Status::OK)
	{
		std::vector<vec4_t> pileup;

		//~ if(pileup_noise2signal > 0.)
		//~ {
			//~ double const pu_totalE_target = pileup_noise2signal * pythia.event.scale();
			//~ double const pu_meanE = pileup_meanF * pythia.event.scale();
			
			//~ double pu_TotalE= 0.;
			//~ double thisE;
			//~ double pu_maxE = -1.;
			
			//~ // while(pu_TotalE < pu_totalE_target) // original stop, for 1 - CDF test below
			//~ while(pu_TotalE < pu_totalE_target)
			//~ {
				//~ vec3_t const pu = IsoVec3_Exponential(gen, pu_meanE, thisE);
				//~ pileup.emplace_back(thisE, pu, kdp::Vec4from2::Energy);
				//~ pileup.emplace_back(thisE, -pu, kdp::Vec4from2::Energy); // Simply correct momentum imbalance
				
				//~ pu_TotalE += 2. * thisE;
				//~ pu_maxE = std::max(pu_maxE, thisE);
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
		
		(*detector)(pythia, pileup);				
				
		//~ auto const finalState_ME = PhatF::PythiaToPhatF(detector.ME());
		detected = detector->Tracks();
		detected.insert(detected.end(), 
			detector->Towers().cbegin(), detector->Towers().cend());
		
		// Transfer pythia to jets	
		{
			auto const finalState_ME = detector->ME();
			
			real_t Etot = 0.;
			
			for(auto const& particle : finalState_ME)
				Etot += particle.e();
				
			for(auto const& particle : finalState_ME)
				ME_vec.emplace_back(particle.px()/Etot, particle.py()/Etot, particle.pz()/Etot,
					particle.m()/Etot, kdp::Vec4from2::Mass);
		}
		
		{
			// Fill the particle into a format useful by fastjet
			std::vector<fastjet::PseudoJet> protojets;
			{
				for(PhatF const& particle : detected)
				{
					vec3_t const p3_scaled = (particle.pHat * particle.f);
					
					// energy goes last (wtf?!)
					protojets.emplace_back(p3_scaled.x1, p3_scaled.x2, p3_scaled.x3, particle.f);
				}
			}
			
			// run the clustering, extract the jets
			fastjet::ClusterSequence cs(protojets, clusterAlg);
			std::vector<fastjet::PseudoJet> const jets = fastjet::sorted_by_E(cs.inclusive_jets());
			
			fast_jets.insert(fast_jets.end(), jets.cbegin(), jets.cend());
		}
			
		////////////////////////////
		
		//~ H_ME = Hcomputer(finalState_ME, lMax);
		//~ H_showered = Hcomputer(detector.FinalState(), lMax);
		H_det = Hcomputer(detected, lMax);
	}
	
	return status;
}

LHE_Pythia_PowerJets::~LHE_Pythia_PowerJets() 
{
	delete detector;
}

LHE_Pythia_PowerJets::vec3_t LHE_Pythia_PowerJets::IsoVec3_Exponential
	(pqRand::engine& gen, double const meanE, double& length) 
{
	// Easiest way to draw an isotropic vec3 is rejection sampling
	
	vec3_t iso(false); // Don't initialize (false)
	double r2;
	
	do
	{
		// U_S has enough precision for this application
		iso.x1 = gen.U_even();
		iso.x2 = gen.U_even();
		iso.x3 = gen.U_even();
		
		r2 = iso.Mag2();
	}
	while(r2 > 1.);
	
	// Now we have a vector whose direction is guarenteed to be isotropic, 
	// but whose length is not only wrong but seems totally useless;
	// we will downscale the vector by its current length when we resize it to
	// a length randomly drawn from the exponential distribution.
	// However, its current length is not entirely useless. 
	// It is in fact a random length drawn from some predictable distribution.
	// So what if we can use it as the input for the quantile function that
	// maps out the exponential distribution. This way we can save 
	// drawing another random number.
	// (NOTE: We will have drawn three random numbers, very much like 
	//  if we had drawn theta, phi, and a random length.
	//  So we can see that we still have to draw the same # of d.o.f.).
	
	// First we must figure out the distribution of r^2 from volume element
	// 	dV / V = (4 pi r**2 dr ) / (4/3 pi R**3)   (R = 1)
	// 	dV / (V dr) = 3 r**2     r2 = r**2    dr2 = 2 r dr
	// 	dV / (V dr2) = 3/2 r2**(1/2)
	// Now we solve for the CDF of r2 (CFD(r2) = \int_{r2p=0}^{r2p=r2}r2p**(3/2)
	// 	u = CDF(r2) = (r2)**(3/2)
	// Now we can plug this uniformly distributed u
	// into the quantile function for an exponential distribution
	//    Q1(u) = -log(u)/lambda    Q2(u) = -log1p(u)/lambda
	// If we want to implement a quantile flip-flop, we simply
	// halve the output of the CDF
	
	{
		double const hu = 0.5 * std::pow(r2, 1.5);
		length = meanE * (gen.RandBool() ? -std::log(hu): -std::log1p(-hu));
	}
					
		// Now scale the vector to the new length (dividing by current)
	iso *= (length / std::sqrt(r2));
		
	// Now move to the other 7 octants
	gen.ApplyRandomSign(iso.x1);
	gen.ApplyRandomSign(iso.x2);
	gen.ApplyRandomSign(iso.x3);
	
	return iso;
}
