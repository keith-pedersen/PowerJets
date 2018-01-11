// A program for calculating the smeared spectral power
// By Keith.David.Pedersen@gmail.com (kpeders1@hawk.iit.edu)
// Adapted from PYTHIA main11.cc (licenced under the GNU GPL version 2)

#include "zLib/kdpVectors.hpp"
//~ #include "pqRand/distributions.hpp"
#include "helperTools.hpp"

#include "fastjet/ClusterSequence.hh"

#include "Pythia8/Pythia.h"
#include "SpectralPower.hpp"
//#include "RandomSmearer.hpp"
#include "ArrogantDetector.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <assert.h>
#include <numeric>
#include <string>

#include <QtCore/QSettings> // qt-devel package (CentOS 7), -lQtCore

using PhatF = SpectralPower::PhatF;
using PhatFvec = SpectralPower::PhatFvec;

bool JetSorter(fastjet::PseudoJet const& lhs, fastjet::PseudoJet const& rhs)
{
	return lhs[0] > rhs[0];
}

// 19 Mar 2017. All errors and leaks are from Pythia and QSettings. My hands are clean.
int main(int argCount, const char** argVector)
{
	/////////////////////////////////////////////////////////////////////
	// My INI file
	
	constexpr char* defaultINIname = "PowerSpectrum.conf";
	std::string const iniFile = ((argCount > 1) ? argVector[1] : defaultINIname);
	
	// Check if the the file exists to read before loading QSettings, 
	// because QSettings will create the file if it doesn't exist.
	if(not std::ifstream(iniFile.c_str()))
	{
		std::cerr << "\nWarning: No configuration file supplied ... using default values.\n\n";
		std::cerr << "Usage: "
			<< "\tProgram settings are passed via a *.conf file using the\n"
			<< "\tINI file standard (wikipedia.org/wiki/INI_file).\n"
			<< "\tThe first and only argument of the program is the *.conf file name\n"
			<< "\tIf no arguments are passed, we look for\n"
			<< "\t\t<" << defaultINIname << ">\n\n";
	}
	
	QSettings parsedINI(iniFile.c_str(), QSettings::NativeFormat);
	
	// Read the main Pythia controls from the INI file
	std::string const lheFile =
		parsedINI.value("main/lheFile", "./unweighted_events.lhe.gz").toString().toStdString();
	
	if(not std::ifstream(lheFile.c_str()))
	{
		std::cerr << "\Error: No LHE file supplied! "
			<< " <" << lheFile << "> does not exist!\n\n";
		return 1;
	}
	
	std::string const outputDir = 
		parsedINI.value("main/outputDir", "./").toString().toStdString();
		
	size_t const lMax = size_t(parsedINI.value("power/lMax", 1024).toInt());
	
	// -1 means do all, as the size_t will become the largest possible number, 
	// and the Pythia loop will end when the LHE file does
	size_t const skipEvents = size_t(parsedINI.value("main/skipEvents", 0).toInt());
	size_t const maxEventIndex = skipEvents + size_t(parsedINI.value("main/maxEvents", -1).toInt());	
	size_t const numThreads = std::min(16lu, size_t(parsedINI.value("main/numThreads", 4).toInt()));

	// Allow for possibility of a few faulty events.
	size_t const nAbort = 10;
	size_t iAbort = 0;
	
	/////////////////////////////////////////////////////////////////////
	// Set up Pythia
			
	// To send a false for printBanner, we have to send the default value for xmlDir.
	Pythia8::Pythia pythia("../xmldoc", false);
	
	// Hard-code in LHE analysis
	pythia.readString("Beams:frameType = 4");
	pythia.readString(("Beams:LHEF = " + lheFile).c_str());
	
	//~ pythia.readString((std::string("Beams:nSkipLHEFatInit = ") + 
		//~ std::to_string(skipEvents)).c_str());
	
	// Pythia seems to choke on LHE e+e- from MadGraph.
	// This flag matches the input energies based upon the output
	pythia.readString("LesHouches:matchInOut = off"); //https://bugs.launchpad.net/mg5amcnlo/+bug/1622747, Stefan Prestel's recommendation
	//pythia.readString("LesHouches:setLeptonMass = 0"); // Take lepton masses from LHE file, doesn't help
		
	// Read all other flags from a pythia card, changed infrequently (for debugging)
	pythia.readFile("Pythia.conf");
	pythia.init();
		
	/////////////////////////////////////////////////////////////////////
	// Set up detecting, smearing and power spectrum
		
	pqRand::engine gen;
	
	// Keep track of particles in the event
	ArrogantDetector_Lepton detector(parsedINI); // The ArrogantDetector handles everything post shower
	
	// Create a computer for the power spectrum H_l
	SpectralPower Hcomputer(parsedINI);
	
	using vec3_t = SpectralPower::vec3_t;
	
	/////////////////////////////////////////////////////////////////////

	// choose a jet definition
	double const R = 0.4;
	fastjet::JetDefinition jet_def(fastjet::antikt_algorithm, R);	
			
	/////////////////////////////////////////////////////////////////////
	// Generate events
		
	// To skip events properly, since we're studying showered shapes, 
	// we actually need pythia.next() to run.
	
	for(size_t iEvent = 0; iEvent < skipEvents; ++iEvent)
	{
		// Generate events, and check whether generation failed.
		if(!pythia.next())
		{
			// If failure because reached end of file then exit event loop.
			if (pythia.info.atEndOfFile()) break;

			// First few failures write off as "acceptable" errors, then quit.
			if (++iAbort < nAbort) continue;
			break;
		}
	}
		
	for(size_t iEvent = skipEvents; iEvent < maxEventIndex; ++iEvent)
	{
		 // Generate events, and check whether generation failed.
		if(!pythia.next())
		{
			// If failure because reached end of file then exit event loop.
			if (pythia.info.atEndOfFile()) break;

			// First few failures write off as "acceptable" errors, then quit.
			if (++iAbort < nAbort) continue;
			break;
		}
		
		detector(pythia);
				
		auto const finalState_ME = PhatF::PythiaToPhatF(detector.ME());
		auto const detected = PhatFvec::Join(detector.Tracks(), detector.Towers());
		
		// Fill the particle into a format useful by fastjet
		std::vector<fastjet::PseudoJet> protojets;
		
		for(PhatF const& particle : detector.Tracks())
		{
			protojets.emplace_back(particle.f, 
				particle.f*particle.pHat.x1, particle.f*particle.pHat.x2, particle.f*particle.pHat.x3);
		}
		
		for(PhatF const& particle : detector.Towers())
		{
			protojets.emplace_back(particle.f, 
				particle.f*particle.pHat.x1, particle.f*particle.pHat.x2, particle.f*particle.pHat.x3);
		}
		
		//~ fastjet::PseudoJet sum(0., 0., 0., 0.);
		//~ for(auto const& protojet : protojets)
			//~ sum += protojet;
			
		//~ printf("%.3e %.3e %.3e %.3e\n", sum[0], sum[1], sum[2], sum[3]);

		// run the clustering, extract the jets
		fastjet::ClusterSequence cs(protojets, jet_def);
		std::vector<fastjet::PseudoJet> jets = cs.inclusive_jets();
		std::sort(jets.begin(), jets.end(), &JetSorter);
		// NOTE fastjet::sorted_by_E doesn't work		
		
		////////////////////////////
			
		auto const H_ME = Hcomputer(finalState_ME, lMax);
		auto const H_showered = Hcomputer(detector.FinalState(), lMax);
		auto const H_det = Hcomputer(detected, lMax);
		
		std::string const fileName = outputDir + std::to_string(iEvent) + "_ME_final_detected.dat";
		std::ofstream file(fileName.c_str(), std::ios::trunc);
		assert(file);
		
		char buffer[1024];
		
		sprintf(buffer, "# %lu   %lu  %lu\n", 
			detector.ME().size(),
			detector.FinalState().size(), 
			detected.size());
		file << buffer;
		
		//~ {
			//~ assert(detector.ME().size() == 6);
			
			//~ std::vector<double> m_t;
			//~ std::vector<double> m_W;
			//~ std::vector<int> index;
			
			//~ for(int i = 0; i < pythia.event.size(); ++i)
			//~ {
				//~ Pythia8::Particle const& particle = pythia.event[i];
				
				//~ // Previously checked for id and decayed, but doesn't count t > t g and W > W A
				//~ if(std::abs(particle.status()) == 22)
				//~ {
					//~ if(std::abs(particle.id()) == 6) // A top
					//~ {
						//~ m_t.push_back(particle.m());
						//~ index.push_back(i);
					//~ }
					//~ else if(std::abs(particle.id()) == 24)
					//~ {
						//~ m_W.push_back(particle.m());
					//~ }
					//~ else printf("%i\n", iEvent);
				//~ }
			//~ }
			
			//~ // Sometimes the t decays directly to b q q
			//~ if(m_W.size() < 2)
			//~ {
				//~ // About 1% of events don't have an identifiable second W.
				//~ // We seemed to have fixed this problem with the second loop 
				//~ printf("%i \n", iEvent);
				
				//~ for(int i = 0; i < pythia.event.size(); ++i)
				//~ {
					//~ Pythia8::Particle const& particle = pythia.event[i];
					
					//~ if((std::abs(particle.id()) == 6)
						//~ and ((particle.daughter2() - particle.daughter1()) + 1 == 3))
					//~ {
						//~ Pythia8::Vec4 pW;
						
						//~ for(int j = particle.daughter1(); j <= particle.daughter2(); ++j)
						//~ {
							//~ Pythia8::Particle const& daughter = pythia.event[j];
							
							//~ if(std::abs(daughter.id()) < 5)
								//~ pW += daughter.p();
						//~ }
						
						//~ m_W.push_back(pW.mCalc());
					//~ }		
				//~ }
			//~ }
			
			//~ assert(m_t.size() == 2);
			//~ sprintf(buffer, "# %.5e %.5e", m_t[0], m_t[1]);
			//~ file << buffer;
						
			//~ for(double const m : m_W)
			//~ {
				//~ sprintf(buffer, " %.5e", m);
				//~ file << buffer;
			//~ }
			
			//~ file << "\n";				
		//~ }
		
		sprintf(buffer, "% 5lu %.16e %.16e %.16e\n", 0, 1., 1., 1.);
		file << buffer;
		
		for(size_t lMinus1 = 0; lMinus1 < Hcomputer.GetSettings().lMax; ++lMinus1)
		{
			sprintf(buffer, "%5lu %.16e %.16e %.16e\n", lMinus1 + 1, 
				H_ME[lMinus1], H_showered[lMinus1], H_det[lMinus1]);
			file << buffer;
		}
		
		file.close();
		
		std::string const fileName_jetList = fileName + ".jetList";
		file.open(fileName_jetList.c_str(), std::ios::trunc);
		
		file << "# Clustered with fastjet::antikt_algorithm R = 0.4, sorted by energy\n";
		
		for(auto const& jet : jets)
		{
			sprintf(buffer, "% .16e % .16e % .16e\n", jet[1], jet[2], jet[3]);
			file << buffer;
		}
		file.close();		
	}// End event loop
	//pythia.stat();
	printf("\n\n");
	return 0;
}

// That transformation is best which transforms least
// In relation to rotation matrices, we want one which is nearly diagonal. 
// This will create the least amount of floating point cancellation
// (because the original vector is mostly unchanged).
// If we're rotating from the north pole (z^) to v^
// * If v^ is in the sourthern hemisphere, we should rotate to -v^,
//   because this will be a more diagonal rotation (especially is v^ is near the south pole). 
//   To undo this, we simply reverse the sign of the rotated vector.
// * If v^ is near the equator, we should find the closest x^/y^ axis
//   and apply a rotation from that axis to z^. This rotation only 
//   involves coordinate swapping and sign changes; no real math.
// This procedure create "the best" rotations. The second step will not be implemented here.

//~ // Smear the incoming vectors, making sure to down-scale the energy fraction appropriately 
//~ std::vector<PhatF> RandomSmearer::Smear(std::vector<PhatF> const& detected)
//~ {
	//~ std::vector<PhatF> smeared;
	//~ smeared.reserve(detected.size()); // Not enough room, but will grow to adequate size in only a few doublings
	
	//~ // For use by smearFactor equation
	//~ double const nParticles = double(detected.size()); 
	
	//~ for(PhatF const& particle : detected)
	//~ {
		//~ // 1. How many copies will we make of this vector
		//~ size_t const nCopies = 
			//~ size_t(smearFactor);
			//~ size_t(std::ceil(smearFactor * std::sqrt(particle.f * nParticles)));
		
		//~ // Conserve energy; split particle's energy equally among its copies
		//~ double const newF = particle.f / double(nCopies);
			
		//~ // The scheme for smearing around unit vector v^ is relatively simple.
		//~ // 1. We generate random smear vectors around the north pole (z^), 
		//~ // because here the quantile function (i.e. random math) is simple.
		//~ // 2. We then calculate the rotation matrix R that maps z^ to v^,
		//~ // then use R to map all the vectors smeared around the north pole to their new location.
		//~ //   2a. The rotation matrix for a large rotation is ill-conditioned
		//~ //       (due to floating point cancellation in terms dealing with z)
			
		//~ // 3. Calculate the rotation matrix.
		//~ // I wish I could make these const after the fact,
		//~ // but I need a bunch of helpers to make them
		//~ kdp::Vec3 rotateX, rotateY, rotateZ;
		//~ {
			//~ // When z is negative, we generate a rotation to 
			//~ // p^'s additive inverse (oppostive vector),
			//~ // then reflect the smeared and rotated vector back across the origin.
		
			//~ kdp::Vec3 const p = ((particle.pHat.x3 < 0.) ? 
				//~ -(particle.pHat) : particle.pHat);
			
			//~ // Solved for rotation matrix via Mathematica:
			//~ //		Simplify[RotationMatrix[{{0, 0, 1}, {x, y, z}}], Assumptions -> {x > 0, y > 0, z > 0}] /. {x^2 + y^2 + z^2 -> 1}
			//~ // Result: {{(y^2 + x^2 z)/(x^2 + y^2), (x y (-1 + z))/(x^2 + y^2), x}, {(x y (-1 + z))/(x^2 + y^2), (x^2 + y^2 z)/(x^2 + y^2), y}, {-x, -y, z}}
			//~ //    Note the z-1 in the offdiagonal xy term (called Rshare here).
			//~ // Since we have a unit vector, z = sqrt(1- x^2 - y2^), 
			//~ // which we can use to transform Rshare into the form here (multiply top and bottom by (1 + z))
							
			//~ double const x2 = kdp::Squared(p.x1);
			//~ double const y2 = kdp::Squared(p.x2);
			//~ double const T = x2 + y2;
			
			//~ double const Rshare = -(p.x1*p.x2)/(p.x3 + 1.);
			
			//~ rotateX.x1 = (y2 + x2*p.x3)/T;
			//~ rotateX.x2 = Rshare;
			//~ rotateX.x3 = p.x1;
			
			//~ rotateY.x1 = Rshare;
			//~ rotateY.x2 = (x2 + y2*p.x3)/T;
			//~ rotateY.x3 = p.x2;
			
			//~ rotateZ.x1 = -p.x1;
			//~ rotateZ.x2 = -p.x2;
			//~ rotateZ.x3 = p.x3;
		//~ }
		
		//~ // Make a bunch of copies
		//~ for(size_t iCopy = 0; iCopy < nCopies; ++iCopy)
		//~ {	
			//~ // Create uninitialized vector (boolean dummy argument)
			//~ kdp::Vec3 smearedVec(false); 
			
			//~ // 1. Generate phi in the most numerically stable way,
			//~ //    by only using sin to make small values, and cos to make large values, 
			//~ //    using an angle uniformly in [0, 4 pi).
			//~ // We currently exclude exact allignement with x, y, and both diagonals, 
			//~ // since we draw U from the exclusive unit interval.
			//~ // Not really a problem; exceedingly rare.
			//~ {
				//~ // Start in octant 1. 
				//~ const double phi = piOver4 * engine.U();
				//~ smearedVec.x1 = std::cos(phi);
				//~ smearedVec.x2 = std::sin(phi);
				
				//~ // Move to octants 4, 5 and 8
				//~ engine.ApplyRandomSign(smearedVec.x1);
				//~ engine.ApplyRandomSign(smearedVec.x2);
				
				//~ // Randomly reflect across y = x, filling octants 2, 3, 6 & 7
				//~ if(engine.RandomChoice())
				//~ {
					//~ double const tmp = smearedVec.x1;
					//~ smearedVec.x1 = smearedVec.x2;
					//~ smearedVec.x1 = tmp;
				//~ }
			//~ }
			
			//~ // 2. generate sinTheta, cosTheta using the quantile function
			//~ //    	u = 1 - cos(theta) = -sigma**2*log(1 - A * y)
			//~ //    then 
			//~ //			sin(theta) = sqrt(u*(2-u))
			//~ //    This gives us a quite precise small value of sin(theta),
			//~ //    and a maximally precise large value of cos(theta)
			//~ //    Now simply rearrange the quantile function for high precision
			//~ //    in the various regimes of cos(theta)
			//~ {
				//~ double sinTheta, cosTheta;
				
				//~ {
					//~ const double y = engine.U();
					
					//~ // Use a quantile flip flop to fill out the small/large angles
					//~ if(engine.RandomChoice())
					//~ {
						//~ // The small angle quantile
						//~ double u = variance * std::log1p(A*y/(1.-A*y));
						
						//~ if(u > 0.5) 
							// If u > 0.5, then cos(theta) is small and we need a different quantile
							// to give us a high-precision cos(theta).
							// Currently unimplemented
						//~ else
						//~ {
							//~ sinTheta = std::sqrt(u*(2.-u));
							//~ cosTheta = 1.-u;
						//~ }
					//~ }
					//~ else
					//~ {
						//~ // The large angle quantile
						//~ // Even though we're in the tail, default by assuming that 
						//~ // theta is still small enough that 1-cos(theta) > 0.5
						//~ double u = - variance*std::log(expVerySmall + A*y);
						//~ // For most cases, expVerySmall is so small it is effectively zero, 
						//~ // and A is effectively 1. When sigma > 20 degree, it starts to have an effect
						
						//~ // Now check for 3 cases
						//~ if(u < 0.5) // The rotation angle is still small
						//~ {
							//~ sinTheta = std::sqrt(u*(2.-u));
							//~ cosTheta = 1.-u;
						//~ }
						//~ else if(u > 1.5) // Very rare, the rotation angle is very large
						//~ {
							//~ double const w = 
								//~ -variance*std::log1p(std::expm1(2./variance)*y);
								
							//~ sinTheta = std::sqrt(-w*(2. + w));
							//~ cosTheta = -1. - w;
						//~ }
						//~ else // Less rare, the rotation angle is moderate
						//~ {
							//~ cosTheta = variance*std::log(expSmall + expLarge*A*y);
							//~ sinTheta = std::sqrt(1. - kdp::Squared(cosTheta));
						//~ }
					//~ }
				//~ }
								
				//~ smearedVec.x1 *= sinTheta;
				//~ smearedVec.x2 *= sinTheta;
				//~ smearedVec.x3 = cosTheta;
			//~ }
			
			//~ // 3. Rotate the smeared vector. Remember whether or not we need to reflect.
			//~ double const reflect = (particle.pHat.x3 < 0.) ? -1. : 1.;
			
			//~ // PhatF will auto-normalize, to correct for rounding error in the rotation
			//~ smeared.emplace_back(
				//~ reflect*(smearedVec.Dot(rotateX)),
				//~ reflect*(smearedVec.Dot(rotateY)),
				//~ reflect*(smearedVec.Dot(rotateZ)), newF);
			
			//~ // Correct for rounding error
			//~ //smeared.back().pHat.Normalize();			
		//~ }
		
		//~ // Validate the rotation apparatus by directly binning the 
		//~ // angle of rotation.
		//~ for(auto itSmeared = smeared.end() - nCopies; itSmeared not_eq smeared.end(); ++itSmeared)
		//~ {
			//~ thetaHisto.Fill(itSmeared->pHat.InteriorAngle(particle.pHat));
		//~ }
	//~ }
	
	//~ return smeared;
//~ }

//~ // Calculates spectral power by binning the values
//~ // Requires pre-computed Legendre edges
//~ class BinnedSpectralPower
//~ {
	//~ private:
		//~ std::vector<std::vector<double>> binEdges;
		//~ std::vector<std::vector<double>> binWeights;
		
		//~ void ReadBinEdgesAndWeights();
				
	//~ public:
		//~ LegendreBinner(std::string legendreFile);
		
		//~ std::vector<double> S_l(std::vector<PhatF> const& sample);
//~ };

//~ std::vector<double> BinnedSpectralPower::S_l(std::vector<PhatF> const& sample)
//~ {
	//~ std::vector<double> Svec;
	//~ Svec.reserve(binEdges.size() + 1); // binEdges doesn't store l=0 ...
	
	//~ // because S_0 = 1, by construction
	//~ Svec.push_back(1.);
	
	//~ std::vector<double> dotVec; // A vector of all the dot products
	//~ std::vector<double> fAccumulated; // The running sum of f (currently unstable/naive sum)
	
	//~ // First calculate the outer product 
	//~ {
		//~ // Use CorrelationPair stuct so std::sort can simultaneously sort 
		//~ // fProduct in the same order as vectorDot
		//~ std::vector<CorrelationPair> correlation;
		//~ correlation.reserve((sample.size()*(sample.size() + 1))/2);
		
		//~ // O(n^2)
		//~ for(auto i = sample.begin(); i not_eq sample.end(); ++i)
		//~ {
			//~ for(auto j = i; j not_eq sample.end(); ++j)
			//~ {
				//~ correlation.emplace_back(*i, *j);
				
				//~ if(i == j)
					//~ correlation.back().fProduct *= 0.5;
			//~ }
		//~ }
			
		//~ // O(n^2 log(n^2)) = O(2 n^2 log(n)) .. dominates the outer product
		//~ std::sort(correlation.begin(), correlation.end());
		
		//~ // Check for instability
		//~ assert(correlation.back().vectorDot <= 1.);
		//~ assert(correlation.front().vectorDot >= -1.);
	
		//~ // Given the sorted dots (with f co-sorted),
		//~ // re-seperate the data because we will be iterating through 
		//~ // the dots a lot, but only touch the fs in the last stage
		
		//~ // To prevent re-adding a lot of f for each S_l, 
		//~ // Store the accumulated sum. Then the sum in each bin is 
		//~ // simply the difference between the edge members
		//~ // fAccumulate stores the total weight to the LEFT of a given dot product
		//~ {
			//~ double fSum = 0.;
		
			//~ dotVec.reserve(correlation.size());
			//~ fAccumulated.reserve(correlation.size() + 1);
		
			//~ for(CorrelationPair const& pair : correlation)
			//~ {
				//~ dotVec.push_back(pair.vectorDot);
				//~ fAccumulated.push_back(fSum);
				//~ fSum += pair.fProduct; // Maybe implement stable add later			
			//~ }
			
			//~ // fSum at this point should equal exactly 1, ensure the difference is small
			//~ assert(std::fabs(fSum - 1.) < 1e-3);
			//~ fAccumulated.push_back(1.);
		//~ }
	//~ }
	
	//~ // We keep track of the bins by storing the index of the
	//~ // first dot to the right of each edge
	//~ // The initial binning is done using the largest available l,
	//~ // which has the highest granularity by definition. 
	//~ // Using the location of known bin edges for the initial bin,
	//~ // we can drastically speed up the binning for smaller l
	//~ std::vector<EdgeToDot> initialBin;
	
	//~ // Do the initial binning
	//~ {
		//~ std::vector<double> const& initialEdges = binEdges.back();
		//~ initialBin.reserve(initialEdges.size());
		
		//~ dotIndex thisDot = dotVec.begin();
		
		//~ // We must stop the search at the second-to-last edge,
		//~ // otherwise we will iterate out-of-bounds in dotVec.
		//~ // We will handle the final edge after the loop
		//~ auto const edgeEnd = initialEdges.end() - 1;
		
		//~ for(auto thisEdge = initialEdges.begin(); thisEdge not_eq edgeEnd; ++thisEdge)
		//~ {
			//~ // Because we have already verified that all dots are less than 1,
			//~ // and the second-to-last edge is less than 1, 
			//~ // there is no way to go outside dotVec
			//~ while(*thisDot < *thisEdge) ++thisDot;
			//~ initialBin.emplace_back(*thisEdge, thisDot);
		//~ }
		
		//~ // We must not dereference this index ... we must be very careful
		//~ initialBin.emplace_back(1., dotVec.end());
		
		//~ assert(initialBin.front().edge == -1.);
		//~ assert(initialBin.front().index == dotVec.begin());
	//~ }
	
	//~ auto theseEdges = binEdges.begin();
	//~ auto theseWeights = binWeights.begin();
	
	//~ size_t l = 1;
	
	//~ //Calculate S_l, starting with S_1, using the initial bin as a guide
	//~ while(theseEdges not_eq binEdges.end() - 1)
	//~ {
		//~ std::vector<dotIndex> thisBin;
		//~ thisBin.reserve(theseEdges->size());
		
		//~ assert(theseEdges->front() == binEdges.back().front());
		//~ auto itInitialBin = initialBin.begin();
		
		//~ auto const edgeEnd = theseEdges->end() - 1;
		//~ for(auto thisEdge = theseEdges->begin(); thisEdge not_eq edgeEnd; ++thisEdge)
		//~ {
			//~ // Find the last edge from the initial bin which is less than thisEdge
			//~ while((itInitialBin + 1)->edge < *thisEdge) ++itInitialBin;
			
			//~ dotIndex thisDot = itInitialBin->index;
			
			//~ while(*thisDot < *thisEdge) ++thisDot;
			//~ thisBin.push_back(thisDot);
		//~ }		
		//~ thisBin.push_back(dotVec.end());
		
		//~ assert(thisBin.front() == dotVec.begin());
		
		//~ std::vector<double> binnedWeight;
		//~ binnedWeight.reserve(theseWeights->size());
		
		//~ // Now calculate the total f in each bin
		//~ double leftWeight = 0.;
		//~ for(auto thisIndex = thisBin.begin() + 1; thisIndex not_eq thisBin.end(); ++thisIndex)
		//~ {
			//~ double const rightWeight = fAccumulated[*thisIndex - dotVec.begin()];
			//~ binnedWeight.push_back(rightWeight - leftWeight);
			//~ leftWeight = rightWeight;
		//~ }
		
		//~ assert(theseWeights->size() == binnedWeight.size());
		
		//~ Svec.push_back(((2.*l++) + 1)*std::inner_product(theseWeights->begin(), 
		//~ theseWeights->end(), binnedWeight.begin(), 0.));
		
		//~ ++theseEdges;
		//~ ++theseWeights;
	//~ }
	
	//~ std::vector<double> initialBinnedWeight;
	//~ initialBinnedWeight.reserve(initialBin.size());
	
	//~ // Now calculate the total f in each bin
	//~ double leftWeight = 0.;
	//~ for(auto thisIndex = initialBin.begin() + 1; thisIndex not_eq initialBin.end(); ++thisIndex)
	//~ {
		//~ double const rightWeight = fAccumulated[thisIndex->index - dotVec.begin()];
		//~ initialBinnedWeight.push_back(rightWeight - leftWeight);
		//~ leftWeight = rightWeight;
	//~ }
	
	//~ Svec.push_back(((2.*l++) + 1)*std::inner_product(initialBinnedWeight.begin(), 
		//~ initialBinnedWeight.end(), binWeights.back().begin(), 0.));
	
	//~ return Svec;
//~ }

//~ void BinnedSpectralPower::ReadBinEdgesAndWeights(std::string const& legendreFile)
//~ {
	//~ std::ifstream precomputed(legendreFile.c_str(), std::ifstream::in);
	
	//~ if(not precomputed)
		//~ throw std::runtime_error("Legendre file does not exist");
	//~ else
	//~ {	
		//~ std::string aline;
		//~ std::stringstream parser;
		
		//~ // Skip the comment and blank lines
		//~ while(getline(precomputed, aline) and (not aline.empty()) 
			//~ and (aline.at(0) == '#'));
		
		//~ // We should exit with a blank line, the next line should be ? = b
		//~ getline(precomputed, aline);
		//~ parser.str(aline);
		
		//~ size_t b;
		//~ parser >> b;
		
		//~ assert(b < 10);
		
		//~ size_t const edgeFactor = size_t(1) << b;
		
		//~ //Fill_l1(edgeFactor);

		//~ for(size_t l = 1; getline(precomputed, aline); ++l)
		//~ {
			//~ bool success = true;
			
			//~ // We enter with aline being the blank line before data
			//~ {
				//~ // Read the l line
				//~ success = getline(precomputed, aline); parser.clear(); parser.str(aline);
				//~ size_t lFound;
				//~ success = (parser >> lFound);
				//~ // ensure l matches our predictions
				//~ assert(lFound == l);
			//~ }
			//~ size_t const numBins = (edgeFactor * l);
			//~ double cache;
			
			//~ {
				//~ success = getline(precomputed, aline); parser.clear(); parser.str(aline);
			
				//~ binEdges.emplace_back();
				//~ auto& thisEdgeVec = binEdges.back();
				//~ thisEdgeVec.reserve(2*numBins + 1); // We MUST reserve correctly, see below
			
				//~ for(size_t edge = 0; edge <= numBins; ++edge)
				//~ {
					//~ success = parser >> cache;
					//~ thisEdgeVec.push_back(cache);
				//~ }
				
				//~ assert(binEdges.back().back() == 0.);
				
				//~ // Reflect the edges across zero (skipping zero)
				//~ // Must be careful here, we're iterating over an object whose
				//~ // size is changing
				//~ for(auto edgeRev = thisEdgeVec.rbegin() + 1; 
					//~ edgeRev not_eq thisEdgeVec.rend(); ++edgeRev)
				//~ {
					//~ thisEdgeVec.push_back(-(*edgeRev));
				//~ }				
			//~ }
			
			//~ {
				//~ success = getline(precomputed, aline); parser.clear(); parser.str(aline);
			
				//~ binWeights.emplace_back();
				//~ auto& thisWeightVec = binWeights.back();
				//~ thisWeightVec.reserve(2*numBins); // We MUST reserve correctly, see below
			
				//~ for(size_t weight = 0; weight < numBins; ++weight)
				//~ {
					//~ success = parser >> cache;
					//~ thisWeightVec.push_back(cache);
				//~ }
				
				//~ assert(binWeights.back().size() == numBins);
				
				//~ // Reflect the weights across zero (this time don't skip any)
				//~ // Must be careful here, we're iterating over an object whose
				//~ // size is changing
				//~ double const sign = double(bool(l & size_t(1)) ? -1. : 1.);
				//~ for(auto weightRev = thisWeightVec.rbegin();
					//~ weightRev not_eq thisWeightVec.rend(); ++weightRev)
				//~ {
					//~ thisWeightVec.push_back(*weightRev*sign);
				//~ }
				
				//~ if(l==3)
				//~ {
					//~ for(auto weight : thisWeightVec)
					//~ {
						//~ printf("%.2f\n", weight);
					//~ }						
				//~ }
			//~ }
			
			//~ if(not success)
				//~ throw std::runtime_error("Legendre file ended unexpectedly");
		//~ }
	//~ }
//~ }

/*
void BinnedSpectralPower::Fill_l1(size_t const edgeFactor)
{	
	// No need to fill l = 0, because S_0 = 1 always
	//~ // l = 0
	//~ {
		//~ // [-1, 1] 
		//~ double const binWidth = 2./double(edgeFactor);
		//~ int const edgeItMax = int(edgeFactor)/2;
		
		//~ binEdges.emplace_back();
		//~ auto& l0edges = binEdges.back();
		//~ l0edges.reserve(edgeFactor + 1);
		
		//~ for(int edge = -edgeItMax; edge <= edgeItMax; ++edge)
		//~ {
			//~ l0edges.push_back(double(edge)*binWidth);
		//~ }
		
		//~ assert(l0edges[edgeItMax] == 0.);
		//~ assert(l0edges.back() = 1.);
		
		//~ // P_0 = 1 everywhere
		//~ binWeights.emplace_back(edgeFactor + 1, binWidth);
	//~ }
	
	// l = 1
	{
		double const binWidth = 1./double(edgeFactor);
		int const edgeItMax = int(edgeFactor);
		
		{
			binEdges.emplace_back();
			auto& l1edges = binEdges.back();
			l1edges.reserve(2*edgeFactor + 1);
			
			for(int edge = -edgeItMax; edge <= edgeItMax; ++edge)
			{
				l1edges.push_back(double(edge)*binWidth);
			}
			
			assert(l1edges[edgeItMax] == 0.);
			assert(l1edges.back() = 1.);
		}
			
		{
			binWeights.emplace_back();
			auto& l1weight = binWeights.back();
			l1weight.reserve(2*edgeFactor);
			
			for(int weight = -edgeItMax; weight < edgeItMax; ++weight)
			{
				l1weight.push_back((double(weight) + 0.5)*kdp::Squared(binWidth));
			}
			
			assert(l1weight.back() == (1.-0.5*binWidth)*binWidth);
		}
	}		
}
*/
