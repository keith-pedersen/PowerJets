#include "LHE_Pythia_PowerJets.hpp"
#include "SpectralPower.hpp"
#include "kdp/kdpTools.hpp"

void LHE_Pythia_PowerJets::ClearCache()
{
	detected.clear();
	detected_PhatF.clear();
	pileup.clear();
	
	Hl_FinalState.clear();
	Hl_Obs.clear();
		
	fast_jets.clear();
	ME_vec.clear();
	
	tracks.clear();
	towers.clear();
}

LHE_Pythia_PowerJets::LHE_Pythia_PowerJets(std::string const& ini_filePath):
	LHE_Pythia_PowerJets(QSettings(ini_filePath.c_str(), QSettings::IniFormat)) {}
	
LHE_Pythia_PowerJets::LHE_Pythia_PowerJets(QSettings const& parsedINI):
	// Because of chained defalt arguments, to send a false for printBanner, 
	// we have to send the default value for xmlDir.
	pythia("../xmldoc", false),
		// Note: Must pass NativeFormat, otherwise it doesn't work.
	detector(ArrogantDetector::NewDetector(parsedINI, "detector")),
		// The detector's parameters are read from the INI folder [detector]
	clusterAlg(), // initializd in main body			
	//~ Hcomputer(parsedINI),
		// HComputer's settings are read from the INI folder [power]
	gen(), // deafult seed
	iEvent_plus1(0),	
	//~ lMax(parsedINI.value("power/lMax", 128).toInt()),
	status(Status::UNINIT),
	trackShape(nullptr), towerShape(nullptr)
{
	// We could be using a file on disk which uses the default name.
	// We will only use all default values when ini_filePath is not on disk.
	//~ if(not std::ifstream(ini_filePath.c_str()))
		//~ std::cerr << "\nWarning: No configuration file supplied ... everything default values.\n\n";
	
	kdp::LimitMemory(2.);
	
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
	
	{
		trackShape = new h_Gaussian(kdp::ReadAngle<double>(
			parsedINI.value("smear/tracks", "1 deg").toString().toStdString()));
		trackShape->hl_Vec(1024lu);
		
		auto towerArea = detector->GetTowerArea();
		std::vector<double> area2;
		
		for(auto const& areaVec : towerArea)
			area2.push_back(areaVec.Mag2());
		
		std::sort(area2.begin(), area2.end()); 
		
		towerShape = new h_Cap(std::sqrt(area2[area2.size() / 2]));
		std::cout << "frac: " << std::sqrt(area2[area2.size() / 2])/(real_t(4)*M_PI) << std::endl;
		towerShape->hl_Vec(1024lu);
	}
	
	////////////////////
	// Initialize pileup
	pileup_meanF = parsedINI.value("pileup/meanF", 1e-2).toDouble(); // Negative value means no pileup
	pileup_noise2signal = parsedINI.value("pileup/noise2signal", -1.).toDouble(); // Negative value means no pileup
	
	{
		std::string puScheme = parsedINI.value("pileup/balancingScheme", "shim").toString().toStdString();
		
		if(puScheme == "back2back")
			puBalancingScheme = PileupBalancingScheme::back2back;
		else if(puScheme == "shim")
			puBalancingScheme = PileupBalancingScheme::shim;
		else
			throw std::runtime_error("LHE_Pythia_PowerJets: pileup balancing scheme \"" + puScheme + "\" not recognized");
	}
}

LHE_Pythia_PowerJets::Status LHE_Pythia_PowerJets::DoWork()
{
	ClearCache();
	
	if(status == Status::OK)
	{
		MakePileup();
		
		(*detector)(pythia, pileup);
		
		detected = detector->Tracks();
		detected.insert(detected.end(), 
			detector->Towers().cbegin(), detector->Towers().cend());
		
		real_t totalE = 0.;
		
		for(auto const& track : detector->Tracks())
		{
			tracks.emplace_back(track);
			totalE += tracks.back().f;
		}
		
		for(auto const& tower : detector->Towers())
		{
			towers.emplace_back(tower);
			totalE += towers.back().f;
		}
							
		for(auto& track : tracks)
			track.f /= totalE;
		
		for(auto& tower : towers)
			tower.f /= totalE;
			
		tracksTowers = ShapedParticleContainer(tracks, *trackShape);
		tracksTowers.append(towers, *towerShape);
			
		//~ detected_PhatF = PhatF::To_PhatF_Vec(detected);
		
		// Transfer pythia ME into jets	
		{
			auto const finalState_ME = detector->ME();
			
			real_t Etot = 0.;
			
			for(auto const& particle : finalState_ME)
				Etot += particle.e();
				
			for(auto const& particle : finalState_ME)
				ME_vec.emplace_back(particle.px()/Etot, particle.py()/Etot, particle.pz()/Etot,
					particle.m()/Etot, kdp::Vec4from2::Mass);
		}
	}
	
	return status;
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
	
	return DoWork();
}

void LHE_Pythia_PowerJets::ClusterJets() const
{
	// Fill the particle into a format useful by fastjet
	std::vector<fastjet::PseudoJet> protojets;
	{
		for(vec3_t const& particle : detected)
		{
			// energy goes last (wtf?!)
			protojets.emplace_back(particle.x1, particle.x2, particle.x3, particle.Mag());
		}
	}
	
	// run the clustering, extract the jets
	fastjet::ClusterSequence cs(protojets, clusterAlg);
	std::vector<fastjet::PseudoJet> const jets = fastjet::sorted_by_E(cs.inclusive_jets());
	
	// We use insert to force an element-wise call to Jet(fastjet::PseudoJet const&)
	fast_jets.insert(fast_jets.end(), jets.cbegin(), jets.cend());
}

void LHE_Pythia_PowerJets::WriteAllVisibleAsTowers(std::string const& filePath)
{
	std::ofstream file(filePath, std::ios::trunc);
	
	// Write to a *.dat with space/tab separated for easy Mathematica import
	if(file.is_open())
	{
		char buffer[1024];
		
		auto const allAsTowers = detector->AllVisibleAsTowers();
		
		for(auto const& tower : allAsTowers)
		{
			sprintf(buffer, "% .6e % .6e % .6e % .6e % .6e\n", 
				tower.first[0], tower.first[1], tower.first[2], tower.first[3],
				tower.second);
			file << buffer;
		}		
	}
}

LHE_Pythia_PowerJets::~LHE_Pythia_PowerJets() 
{
	delete detector;
	delete trackShape;
	delete towerShape;
}

LHE_Pythia_PowerJets::vec4_t LHE_Pythia_PowerJets::IsoVec3_Exponential
	(pqRand::engine& gen, real_t const meanE) 
{
	// Easiest way to draw an isotropic vec3 is rejection sampling
	
	vec4_t iso(false); // Don't initialize (false)
	real_t r2;
	
	do
	{
		// U_S has enough precision for this application
		iso.x1 = gen.U_even();
		iso.x2 = gen.U_even();
		iso.x3 = gen.U_even();
		
		r2 = iso.p().Mag2();
	}
	while(r2 > real_t(1));
	
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
		real_t const hu = real_t(0.5) * std::pow(r2, real_t(1.5));
		real_t const energy = meanE * (gen.RandBool() ? -std::log(hu): -std::log1p(-hu));
	
		// Now scale the vector to the new length (dividing by current)
		iso *= (energy / std::sqrt(r2));
		iso.x0 = energy;
	}
		
	// Now move to the other 7 octants
	gen.ApplyRandomSign(iso.x1);
	gen.ApplyRandomSign(iso.x2);
	gen.ApplyRandomSign(iso.x3);
	
	return iso;
}

std::vector<Jet> const& LHE_Pythia_PowerJets::Get_FastJets() const
{
	if(fast_jets.empty()) 
		ClusterJets();
	
	return fast_jets;
}


std::vector<LHE_Pythia_PowerJets::real_t> const&
LHE_Pythia_PowerJets::Get_Hl_FinalState(size_t const lMax)
{
	if(Hl_FinalState.size() <= lMax)
		Hl_FinalState = Hl_computer.Hl_Obs(lMax,
			ShapedParticleContainer(PhatF::To_PhatF_Vec(detector->FinalState()), *trackShape));
	return Hl_FinalState;
}

std::vector<LHE_Pythia_PowerJets::real_t> const&
LHE_Pythia_PowerJets::Get_Hl_Obs(size_t const lMax)
{
	if(Hl_Obs.size() < lMax)
		Hl_Obs = Hl_computer.Hl_Obs(lMax, tracksTowers);
	return Hl_Obs;
}

std::vector<LHE_Pythia_PowerJets::real_t>
LHE_Pythia_PowerJets::Get_Hl_Obs_slow(size_t const lMax)
{
	return SpectralPower::Hl_Obs(lMax, tracks, *trackShape, towers, *towerShape);
}

std::vector<LHE_Pythia_PowerJets::real_t>
LHE_Pythia_PowerJets::Get_Hl_Jet(size_t const lMax, std::vector<ShapedJet> const& jets)
{
	return Hl_computer.Hl_Jet(lMax, jets, Get_DetectorFilter(lMax));
}

std::vector<LHE_Pythia_PowerJets::real_t>
LHE_Pythia_PowerJets::Get_Hl_Jet_slow(size_t const lMax, std::vector<ShapedJet> const& jets)
{
	return SpectralPower::Hl_Jet(lMax, jets, Get_DetectorFilter(lMax));
}

std::vector<LHE_Pythia_PowerJets::real_t>
LHE_Pythia_PowerJets::Get_Hl_Hybrid(size_t const lMax, std::vector<ShapedJet> const& jets)
{
	return Hl_computer.Hl_Hybrid(lMax, jets, Get_DetectorFilter(lMax), 
		tracksTowers, Get_Hl_Obs(lMax));
}

std::vector<LHE_Pythia_PowerJets::real_t>
LHE_Pythia_PowerJets::Get_Hl_Hybrid_slow(size_t const lMax, std::vector<ShapedJet> const& jets)
{
	return SpectralPower::Hl_Hybrid(lMax, jets, Get_DetectorFilter(lMax),
		tracks, *trackShape, towers, *towerShape, Get_Hl_Obs(lMax));
}

std::vector<LHE_Pythia_PowerJets::real_t> const& LHE_Pythia_PowerJets::Get_DetectorFilter(size_t const lMax)
{
	if(detectorFilter.size() < lMax)
	{
		std::vector<real_t> hl_track = trackShape->hl_Vec(lMax);
		std::vector<real_t> hl_tower = towerShape->hl_Vec(lMax);
		
		hl_track *= chargeFraction;
		hl_tower *= real_t(1) - chargeFraction;
		
		detectorFilter = hl_track + hl_tower;
	}
	
	return detectorFilter;
}

void LHE_Pythia_PowerJets::MakePileup()
{
	pileup.clear();
	
	if(pileup_noise2signal > real_t(0))
	{
		real_t const pu_totalE_target = pileup_noise2signal * pythia.event.scale();
		real_t const pu_meanE = pileup_meanF * pythia.event.scale();
		
		assert(pu_meanE > real_t(0));
		
		real_t pu_TotalE = real_t(0);
		real_t pu_maxE = real_t(-1);
						
		// Ensuring that pileup sums to zero is not trivial.
		// Adding up isotropic 3-vectors creates a 3D random walk, 
		// so while we expect (sum/n) to converge to zero, (sum) itself will diverge.
		// I have found 2 methods which keep sum balanced:
		//    1. Quick and dirty; draw a vector, add its opposite.
		// 	2. Slightly less dirty; draw 2 vectors, add the opposite of their sum.
		//		3. Monitor sum and whenever sum.Mag() gets too large,
		//			"shim" it by adding a unit vector opposite of sum. 
		//       Leave space for 2 unit vectors to neutralize the final sum.
		// This latter was designed for isotropic unit vectors, so we will not use it.
		// A quick study in Mathematica (ExponentialPileupBalancing) shows that
		// method #2 does not egregiously alter the exponential distribution,
		// only shifting up it's mean by about 10%
		// (and slightly diminishing the probability of zero-energy particles).
		
		switch(puBalancingScheme)
		{
			case PileupBalancingScheme::back2back:
				while(pu_TotalE < pu_totalE_target)
				{
					pileup.push_back(IsoVec3_Exponential(gen, pu_meanE));
					pileup.emplace_back(pileup.back().x0, -pileup.back().p(), kdp::Vec4from2::Energy);
					
					pu_TotalE += 2. * pileup.back().x0;
					pu_maxE = std::max(pu_maxE, pileup.back().x0);
					
					//~ if(pileup.size() > (size_t(1) << 10))
						//~ throw std::runtime_error("problem");
				}
			break;
			
			case PileupBalancingScheme::shim:
				while(pu_TotalE < pu_totalE_target)
				{
					vec3_t sum;
					
					for(size_t i = 0; i < 2; ++i)
					{
						pileup.push_back(IsoVec3_Exponential(gen, pu_meanE));
						sum += pileup.back().p();
						
						pu_TotalE += pileup.back().x0;
						pu_maxE = std::max(pu_maxE, pileup.back().x0);
					}
					
					pileup.emplace_back(0., -sum, kdp::Vec4from2::Mass);
					pu_TotalE += pileup.back().x0;
					pu_maxE = std::max(pu_maxE, pileup.back().x0);
				}
			break;
		}
			
		// THe lesson below: Added naively, pileup's random walk 
		// creates a large momentum imbalance which cannot be accounted for 
		// by the addition of a single particle opposite the total, 
		// because the energy of that single particle is way too large.
		// The results of the first few events examined:
		// 1-CDF: {1.886e-03, 3.987e-04, 8.038e-05, 1.551e-05, 1.762e-04, 5.858e-03, 9.268e-08}
		//
		// CDF: 1 − e−λx
		// λ = 1 / pu_meanE
		// double const puTotalP = std::accumulate(pileup.begin(), pileup.end(), vec3_t()).Mag();
		// printf("1-CDF: %.3e\n", std::exp(-puTotalP/pu_meanE));
	}
}
