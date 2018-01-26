// A program for calculating the smeared spectral power
// By Keith.David.Pedersen@gmail.com (kpeders1@hawk.iit.edu)
// Adapted from PYTHIA main11.cc (licenced under the GNU GPL version 2)

#include "SpectralPower.hpp"
#include "ArrogantDetector.hpp"
#include "NjetModel.hpp"

#include "fastjet/ClusterSequence.hh"

#include "Pythia8/Pythia.h"

#include <vector>
#include <string>
#include <sstream>

#include <QtCore/QSettings> // qt-devel package (CentOS 7), -lQtCore

class LHE_Pythia_PowerJets
{
	public:
		using real_t = SpectralPower::real_t;
		using PhatF = SpectralPower::PhatF;
		using PhatFvec = SpectralPower::PhatFvec;

		using vec3_t = ArrogantDetector::vec3_t;
		using vec4_t = ArrogantDetector::vec4_t;
		
		enum class Status {OK, UNINIT, END_OF_FILE, EVENT_MAX, ABORT_MAX};
		
		static constexpr char* ini_filePath_default = "PowerJets.conf";
		
	private:
		Pythia8::Pythia pythia; //! @brief The Pythia instance
		
		QSettings parsedINI; //! @brief The parsed ini file, used to control everything
		ArrogantDetector* detector; //! @brief The detector
			// This is a pointer so we can have either lepton or hadron detector
			// In the future we can add something like Delphes
		fastjet::JetDefinition clusterAlg; //! @brief FastJet's clustering algorithm
		SpectralPower Hcomputer; //! @brief calculates spectral power
		pqRand::engine gen; //! A PRNG
		
		size_t iEvent_plus1;
		size_t iEvent_end;
		size_t abortsAllowed;
		size_t abortCount;
		size_t lMax;
		Status status;
		
		// For now, we only cache the pieces we need
		std::vector<PhatF> detected; // The final state particles
		std::vector<real_t> H_det; // The power spectrum for detected particles
		std::vector<Jet> fast_jets; // Jest clustered from particle_cache using Fastjet
		
		/*! @brief A place to redirect the FastJet banner.
		 * 
		 *  \warning This must be removed if this code is distributed, per the GPL license.
		*/ 
		std::stringstream fastjet_banner_dummy;
		
		void ClearCache();
		
		Status Next_internal(bool skipAnal);
		
	public:
		LHE_Pythia_PowerJets(std::string const& ini_filePath = ini_filePath_default);
		
		~LHE_Pythia_PowerJets();
		
		//////////////////////////////////////////////////////////////////
		
		size_t EventIndex() {return iEvent_plus1 - 1;}
		
		Status GetStatus() {return status;}
		
		std::vector<PhatF> const& GetDetected() {return detected;}
		
		std::vector<real_t> const& Get_H_det() {return H_det;} 
		
		std::vector<Jet> const& Get_FastJets() {return fast_jets;}
		
		Status Next() {return Next_internal(false);}

		// Give me a vector pointing in an isotropic direction whose 
		// length follows an exponential distribution with mean = 1/lambda
		static vec3_t IsoVec3_Exponential(pqRand::engine& gen, double const meanE, double& length);
};

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
