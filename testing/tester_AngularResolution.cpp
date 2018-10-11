// Files from the PowerSpectrum package
#include "PowerSpectrum.hpp"
#include "ArrogantDetector.hpp"

// Files from the kdp package (which provides the vector math for PowerJets)
#include "kdp/kdpTools.hpp"
#include "kdp/kdpSettings.hpp"

#include "pqRand/pqRand.hpp"

#include "Pythia8/Pythia.h"

#include <string>
#include <vector>
#include <fstream> // ifstream
#include <iostream> // cout
#include <exception> // runtime_error
#include <memory> // unique_ptr
#include <algorithm> // accumulate
#include <chrono>
#include <random>


// A popular library with a nice INI file parser
#include <QtCore/QSettings>

// A quick and easy settings class designed to work with QSettings
// New settings are easily added, with their key's and default value.
// Remember to add a Read line to the ctor.

// Debugging information (-g) takes no time
// Assertions take a minimal amount of time
// PowerSpectrum with shape takes about twice as long
// This actually makes sense, because ultimately we have a total sum over 
// [f_i * f_j] x [P_l(p^_i . p^_j)] x [h_l_i * h_l_j]
// So we have 3 outer products, two of which are l-dependent.
// When we don't have shape, the time is limited by one l-dependent outer product
// Adding shape, we get two l-dependent outer products, effectively doubling the time
struct Settings : public kdp::Settings_Base
{
	// Number of events to run
	Param<size_t> nEvents = Param<size_t>("nEvents", 100);
	
	// Number of events per detection
	Param<size_t> overlap_max = Param<size_t>("overlap_max", 20);
	
	// How many threads to use calculating H_l
	Param<size_t> nThreads = Param<size_t>("nThreads", 4);
	
	Param<size_t> skipEvents = Param<size_t>("skipEvents", 0);
	
	Param<double> relError_max = Param<double>("relError_max", 0);
			
	// Given a parsed INI file, read in the settings
	Settings(QSettings const& parsedINI)
	{
		nEvents.Read(parsedINI);
		overlap_max.Read(parsedINI);
		nThreads.Read(parsedINI);
		skipEvents.Read(parsedINI);
		relError_max.Read(parsedINI);
	}
};

using kdp::Vec4;

int main(int argCnt, char const** const argVec)
{
	// These simulations can grow out of hand fast (leading to
	// uninterruptable swap trashing). Kill the program if it uses more than 2 GB
	kdp::LimitMemory(4.);
	
	/////////////////////////////////////////////////////////////////////
	// If an argument is given, it is the path to the INI file
	std::string const iniPath = (argCnt > 1) ? 
		argVec[1] : "tester_AngularResolution.ini";
		
	if((argCnt > 1) and (not std::ifstream(iniPath).is_open())) // Make sure there is an INI file
	{
		std::cerr << ("INI file <" + iniPath + "> cannot be read.\n");
		return 1;
	}
	
	/////////////////////////////////////////////////////////////////////
	// Create a settings object that parses the INI file
	QSettings parsedINI(iniPath.c_str(), QSettings::IniFormat);
	Settings settings(parsedINI);
	
	size_t const iEvent_end = std::max(settings.nEvents.value, 
		settings.skipEvents + settings.nEvents);
	
	// Create a PowerSpectrum thread pool for fast calculation
	PowerSpectrum Hl_calculator(settings.nThreads);
	
	// Read detector settings from the INI file
	std::unique_ptr<ArrogantDetector> detector(ArrogantDetector::NewDetector(parsedINI, "Detector"));
	
	/////////////////////////////////////////////////////////////////////
	// Create a Pythia instance and initialize it to simple processes
	Pythia8::Pythia pythia;
	
	pythia.readString("Beams:frameType = 1"); // CM frame ...
	pythia.readString("Beams:eCM = 14000"); // ... so only need CM energy
	
	// e+ e- machine
	pythia.readString("Beams:idA = 11");
	pythia.readString("Beams:idB = -11");
	pythia.readString("PDF:lepton = off ");
	
	// Turn on LO QCD; focus on hadronic final state only
	pythia.readString("WeakSingleBoson:ffbar2gmZ = on");
	pythia.readString("23:onMode = off");
	pythia.readString("23:onIfAny = 1 2 3 4 5");
	
	// Suppress a bunch of output
	pythia.readString("SLHA:readFrom = 0"); // Don't worry about SUSY, otherwise we print a bunch of SUSY settings
	
	pythia.readString("Init:showMultipartonInteractions = off");
	pythia.readString("Init:showChangedSettings = off");
	pythia.readString("Init:showProcesses = off");
	pythia.readString("Init:showChangedParticleData = off");
	
	pythia.readString("Next:numberShowEvent = 0");
	pythia.readString("Next:numberShowProcess = 0");
	pythia.readString("Next:numberShowInfo = 0");
	pythia.readString("Next:numberShowLHA = 0");
	
	pythia.init();
	size_t abortCount = 0;
		
	/////////////////////////////////////////////////////////////////////
	// Instantiate a PRNG
	pqRand::engine gen(false); // don't initialize
	gen.Seed_Reuse("pqRand.seed");
	
	std::uniform_int_distribution<size_t> n_overlap(0, settings.overlap_max);
	
	for(size_t iEvent = 0; iEvent < iEvent_end; ++iEvent)
	{
		detector->Clear(); // Clear the detector of all energy
		size_t const overlap = n_overlap(gen);
		
		{
			for(size_t i = 0; i <= overlap; ++i)
			{
				// Generate the next event; catch Pythia aborts and retry.
				while(not pythia.next())
				{
					if(++abortCount == 10)
					{
						std::cerr << ("Pythia aborted too many times.\n");
						return 1;
					}
				}
				
				if(iEvent >= settings.skipEvents)
					detector->PartialFill(pythia); // Fill some energy in
			}
		}
				
		if(iEvent >= settings.skipEvents)
		{
			detector->Finalize(); // no argument means no missing E correction
			
			auto const observation = detector->GetObservation();
			
			auto t1 = std::chrono::high_resolution_clock::now();
			double const angular_res = kdp::ToDegrees(PowerSpectrum::AngularResolution_Slow(observation));
			auto t2 = std::chrono::high_resolution_clock::now();
			
			double const time_orig = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
			
			t1 = std::chrono::high_resolution_clock::now();
			double const angular_res_new = kdp::ToDegrees(Hl_calculator.AngularResolution(observation));
			t2 = std::chrono::high_resolution_clock::now();
			
			double const time_new = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();		
			double const relError = kdp::AbsRelError(angular_res_new, angular_res);
					
			printf("event: %3lu; overlap: %3lu; %5lu tracks; %5lu towers; ang. res = %.3e deg; rel. error = %.3e; speedup = %.1f\n",
				iEvent, overlap,
				observation.tracks.size(), observation.towers.size(), 
				angular_res, relError, time_orig / time_new);
				
			if(relError > settings.relError_max)
			{
				printf("FAIL!\n");
				return 1;
			}
		}
	}
	
	printf("PASS.\n");
	return 0;
}
