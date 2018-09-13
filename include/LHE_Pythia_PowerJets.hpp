#ifndef LHE_PYTHIA_POWERJETS
#define LHE_PYTHIA_POWERJETS

// Copyright (C) 2018 Keith Pedersen (Keith.David.Pedersen@gmail.com)
// Adapted from PYTHIA main11.cc and main19.cc (licenced under the GNU GPL version 2)

#include "PowerJets.hpp"
#include "PowerSpectrum.hpp"
#include "ArrogantDetector.hpp"
#include "NjetModel.hpp"
#include "ShapeFunction.hpp"

#include "fastjet/AreaDefinition.hh"
#include "fastjet/Selector.hh"
#include "fastjet/tools/JetMedianBackgroundEstimator.hh"
#include "fastjet/tools/Subtractor.hh"

#include <Pythia8/Pythia.h>

#include <vector>
#include <string>

#include <QtCore/QSettings> // qt-devel package (CentOS 7), -lQtCore

/*! @file LHE_Pythia_PowerJets.hpp
 *  @brief Defines an object which runs Pythia on an LHE file and 
 *  analyzes the final state using the PowerJets package.
 *  @author Copyright (C) 2018 Keith Pedersen (Keith.David.Pedersen@gmail.com, https://wwww.hepguy.com)
*/

GCC_IGNORE_PUSH(-Wpadded)

/*! @brief A Pythia generator which runs from an LHE file and 
 *  analyzes the showered, hadronized final state using the PowerJets package.
 *  
 *  Pythia is a Monte Carlo simulator for particle collisions and jet formation. 
 *  A common use case has an external program (such as MadGraph) 
 *  outputting a collection of parton-level collisions (the "matrix element") into an LHE file.
 *  Pythia can then read the LHE and dress the matrix element into a 
 *  more complete physics process (e.g., simulate jet formation to detectable particles, 
 *  initial state radiation, multi-parton interactions, pileup, etc.).
 * 
 *  Once LHE_Pythia_PowerJets is constructed, Next() must be called at least once.
 *  Each time Next() is called, a new event is generated and analyzed.
 * 
 *  The LHE_Pythia_Pileup is controlled via an INI file using keys in named sections:
 *   
 *   - Main: General control (e.g. the title of the detector section in the INI file)
 * 
 *   - Pythia: How this object runs Pythia (e.g. the location of the LHE file).
 * 
 *   - PowerJets: To calculate an accurate power spectrum, 
 *     charged particles are smeared using h_PseudoNormal with a lambda 
 *     determined by the following formula: a circular cap of 
 *     radius \f$ \in R [0, \pi] \f$ contains a fraction \f$ u \in (0, 1] \f$ 
 *     of the particle's spatial probability distribution.
 *     R is determined from the sample's overall angular resolution.
 * 
 *   - FastJet: The final state is clustered using FastJet's inclusive clustering algorithms.
 * 
 *   - Pileup: A second Pythia generator is used to generate LHC-like pileup,
 *     which can be added to each event. It is assumed that all charged 
 *     pileup which \em can be tracked (within the tracker's \f$ \eta \f$ coverage
 *     and with enough transverse momentum) \em is tracked to a vertex 
 *     distinguishable from the primary vertex (and thus can be removed from the hard scatter).
 *     Neutral pileup cannot be subtracted.
 * 
 *  LHE_Pythia_PowerJets::Settings explains the available keys in detail
*/  
class LHE_Pythia_PowerJets
{
	public:
		using real_t = PowerJets::real_t;
		using PhatF = PowerSpectrum::PhatF;
		using VecPhatF = PowerSpectrum::VecPhatF;
		using PhatFvec = PowerSpectrum::PhatFvec;
		using ShapedParticleContainer = PowerSpectrum::ShapedParticleContainer;
		using DetectorObservation = PowerSpectrum::DetectorObservation;

		using vec3_t = ArrogantDetector::vec3_t; //! @brief The 3-vector type
		using vec4_t = ArrogantDetector::vec4_t; //! @brief The 4-vector type
		
		//! @brief A strongly-typed enum to communicate the state of the internal Pythia generator
		enum class Status {OK //! @brief Everything is OK
			, UNINIT //! @brief Pythia has not run yet; call Next()
			, END_OF_FILE //! @brief We have reached the end of the LHE file, no more events to generate
			, EVENT_MAX  //! @brief We hit the maximum number of events and stopped generating
			, ABORT_MAX //! @brief Pythia aborted too many times ... something's wrong
			};
		
		//! @brief The default location of the INI file
		static constexpr char const* const INI_filePath_default = "PowerJets.ini";
		
		/*! @brief The default location of Pythia's main particle settings
		 * 
		 *  This is the first argument sent to the Pythia constructor, 
		 *  and this string its default value. In order to send true/false 
		 *  for the second argument (printBanner), we store the default value.
		*/		
		static constexpr char const* const Pythia_xmlDoc = "../share/Pythia8/xmldoc";
		
		//! @brief Average jet energy fraction carried by charged particles (measured at LHC)
		static constexpr real_t chargeFraction = real_t(0.59);
		
		/*! @brief The settings used by LHE_Pythia_PowerJets
		 * 
		 *  This class is designed to work in concert with a QSettings INI file, 
		 *  which uses "keys", in sections, to define values.
		 *  Each settings is initilialized by a Param<T>(key, default_value)
		*/ 
		struct Settings : public kdp::Settings_Base
		{
			///////////////////////////////////////////////////////////////
			// Main
			///////////////////////////////////////////////////////////////
			/*! @brief The name of the INI section which sets the detector parameters
			 * 
			 *  This section name is passed to ArrogantDetector::NewDetector, 
			 *  which reads all detector settings from this section.
			*/
			Param<std::string> Main__detectorName = Param<std::string>("Main/detectorName", "Detector");
			
			//! @brief Print the banners of the embedded programs
			Param<bool> Main__printBanners = Param<bool>("Main/printBanners", true);
			
			///////////////////////////////////////////////////////////////
			// Pythia
			///////////////////////////////////////////////////////////////
			
			/*! @brief The path to the LHE file
			 * 
			 *  The LHE file can also be controlled by setting Beams:LHEF = \p filePath
			 *  in the Pythia conf file (which will override this setting).
			*/ 
			Param<std::string> Pythia__lhePath = Param<std::string>("Pythia/lhePath", "./unweighted_events.lhe.gz");
			
			//! @brief How many events to skip
			Param<size_t> Pythia__skipEvents = Param<size_t>("Pythia/skipEvents", 0);
			
			/*! @brief The maximum number of events to generate before Next() has no effect
			 * 
			 *  The default value is -1, which creates the largest possible size_t, 
			 *  so that generation continues until the LHE file runs dry
			*/ 
			Param<size_t> Pythia__maxEvents = Param<size_t>("Pythia/maxEvents", -1);
			
			//! @brief The maximum number of Pythia aborts to allow before stopping
			Param<size_t> Pythia__abortsAllowed = Param<size_t>("Pythia/abortsAllowed", 10);
			
			//! @brief The filepath for Pythia.conf file that controls all other Pythia settings (e.g., tunes, MPI = on, etc)
			Param<std::string> Pythia__confPath = Param<std::string>("Pythia/confPath", "./Pythia.conf");
			
			///////////////////////////////////////////////////////////////
			// PowerJets
			///////////////////////////////////////////////////////////////
			
			//! @brief The track radius R is this factor (fR) multiplied by the sample's angular resolution 
			Param<double> PowerJets__fR_track = Param<double>("PowerJets/fR_track", 1.);
			
			//! @brief The fraction u of each track contained within a cap of radius R
			Param<double> PowerJets__u_track = Param<double>("PowerJets/u_track", 0.9);
			
			///////////////////////////////////////////////////////////////
			// FastJet
			///////////////////////////////////////////////////////////////
			
			//! @brief The clustering algorithm ("kt", "CA" (Cambridge-Aacen), "anti-kt")
			Param<std::string> FastJet__algo = Param<std::string>("FastJet/algo", "anti-kt");
			
			//! @brief The clustering radius
			Param<double> FastJet__R = Param<double>("FastJet/R", 0.4);
			
			//! @brief The clustering algorithm ("kt", "CA" (Cambridge-Aacen), "anti-kt")
			//! for pileup density estimation
			Param<std::string> FastJet__algo_pileup = Param<std::string>("FastJet/algo_pileup", "kt");
						
			//! @brief The clustering radius for pileup density estimation
			Param<double> FastJet__R_pileup = Param<double>("FastJet/R_pileup", 0.4);
			
			//! @brief The number of hardest jets to exclude from pileup density estimation
			Param<size_t> FastJet__nHardExclude_pileup = Param<size_t>("FastJet/nHardExclude_pileup", 2);		
			
			///////////////////////////////////////////////////////////////
			// Pileup
			///////////////////////////////////////////////////////////////
			
			/*! @brief The exact number of pileup vertices.
			 * 
			 *  This is a placeholder for the mean of a Poisson distribution, 
			 *  which will be implemented eventually.
			*/ 
			Param<double> Pileup__mu = Param<double>("Pileup/mu", 0.);
			
			/*! @brief The file path for a separate Pythia.conf file to 
			 *  control pileup generation.
			 * 
			 *  It should at least set "SoftQCD:all = on" and "Beams:eCM = 2 * SomeEnergy"
			*/ 
			Param<std::string> Pileup__confPath = Param<std::string>("Pileup/confPath", "./Pileup.conf");
			
			/*! @brief The file path for a file storing the pileup "up" coefficient
			 *  measured in a prior experiment on the same detector and pileup configuration.
			 * 
			 *  \warning The coefficients are read verbatim; the user shall ensure 
			 *  that this file corresponds exactly to the same detector and Pileup.conf.
			*/ 
			Param<std::string> Pileup__hlPath = Param<std::string>("Pileup/hlPath", "");
			
			Settings(QSettings const& parsedINI);
			~Settings() {}
		};
		
	private:
		Settings settings; //!< @brief Settings to control the generator
	
		//////////////////////////////////////////////////////////////////
		// Pythia
		//////////////////////////////////////////////////////////////////
	
		Pythia8::Pythia pythia; //!< @brief The Pythia instance
		/*! @brief A secondary (optional) Pythia instance, used for pileup
		 * 
		 *  We use a pointer to avoid creating an unused object
		*/ 
		Pythia8::Pythia* pileup;
		
		//////////////////////////////////////////////////////////////////
		// Detector
		//////////////////////////////////////////////////////////////////
		
		/*! @brief The detector
		 *  
		 *  This is a pointer so we can have either 
		 *  ArrogantDetector_Lepton or ArrogantDetector_Hadron.
		*/ 
		ArrogantDetector* detector;
		
		//////////////////////////////////////////////////////////////////
		// PowerSpectrum
		//////////////////////////////////////////////////////////////////
		
		mutable PowerSpectrum Hl_computer; //!< @brief A thread pool to calculate the power spectrum
		
		//////////////////////////////////////////////////////////////////
		// FastJet
		//////////////////////////////////////////////////////////////////
		
		//! @brief The clustering algorithm for the hard scatter
		fastjet::JetDefinition clusterAlg;
		//! @brief The clustering algorithm for the pileup density estimator
		fastjet::JetDefinition clusterAlg_pileup;
		
		//! @brief The jet area definition for the hard scatter
		fastjet::AreaDefinition areaDef;
		//! @brief The jet area definition for the pileup density estimator
		fastjet::AreaDefinition areaDef_pileup;
		
		//! @brief To select which jets are used for the pileup density estimator
		fastjet::Selector pileupSelector;
		//! @brief To estimate the average pileup density
		mutable fastjet::JetMedianBackgroundEstimator pileupEstimator;
		//! @brief To subtract pileup from jets, based upon their area
		fastjet::Subtractor pileupSubtractor;		
				
		//////////////////////////////////////////////////////////////////
		// Bookkeeping
		//////////////////////////////////////////////////////////////////
		
		size_t nextCount; //!< @brief The number of times pythia.next() has been called
		size_t nextCount_max; //!< @brief The maximum nextCount before Next() has no effect
		size_t abortCount; //!< @brief Count of pythia aborts (failed events)
		Status status; //!< @brief The status of the generator		
		
		//////////////////////////////////////////////////////////////////
		// Cached detection
		//////////////////////////////////////////////////////////////////
		
		//~ DetectorObservation observation; //!< @brief The raw detector observation
		ShapedParticleContainer tracksTowers; //!< @brief The extensive tracks and towers
				
		// For now, we only cache the pieces we need
		std::vector<vec3_t> detected; //!< @brief The final state particles
		mutable std::vector<real_t> Hl_Obs; //!< @brief The power spectrum of detector particles (in the lab frame)
		//~ mutable std::vector<real_t> detectorFilter; //!< @brief The "up" coefficient for the detector
		std::shared_ptr<ShapeFunction> pileupShape;
		
		// Original, isotropic pileup
		//~ pqRand::engine gen; //!< @brief A PRNG
		
		//~ enum class PileupBalancingScheme {back2back, shim};
		//~ real_t pileup_noise2signal;
		//~ real_t pileup_meanF;		
		//~ PileupBalancingScheme puBalancingScheme;		
		
		//~ void MakePileup();
		
		/*! @brief A place to redirect the FastJet banner.
		 * 
		 *  \warning This must be removed if this code is distributed, per the GPL license.
		*/ 
		std::stringstream fastjet_banner_dummy;
		
		//////////////////////////////////////////////////////////////////
		// Methods
		//////////////////////////////////////////////////////////////////
		
		void Clear(); //!< @brief Clear all caches
		
		/*! @brief Call pythia.Next(), check for errors, then "do work" if requested
		 * 
		 *  If \p doWork is false, no attempt is made to detect or analyze the final state.
		 *  This is used to skip the first events in an LHE file
		 *  while still running the skipped events through Pythia so that 
		 *  its PRNG is called the same number of times.
		 *  This ensures that event_10 will look the same whether or not 
		 *  event_0 through event_9 are analyzed.
		*/  
		Status Next_internal(bool doWork);		
		Status DoWork(); //!< @brief Detect and analyze the event
		
		//! @brief Initialize a FastJet clustering algorithm and pileup subtractor
		void Initialize_FastJet();
		
		//! @brief Create and initialize the Pythia instances that generates pileup
		void Initialize_Pythia();
		
		/*! @brief Initialize the Pythia generator responsible for pileup
		 * 
		 *  \note This takes some time. If settings.Pileup__mu <= 0 or the
		 *  pileup generator has already been initialized, this function has no effect.
		*/ 
		void Warmup_Pileup();
		
		//!< @brief Cluster the final state with FastJet
		std::vector<Jet> ClusterJets(bool const subtractPileup = false) const;
		
	public:
		//! @brief Construct a QSettings object and call the other constructor
		LHE_Pythia_PowerJets(std::string const& INI_filePath = INI_filePath_default);
		
		//! @brief Read settings from the INI file and initialize the generators
		LHE_Pythia_PowerJets(QSettings const& parsedINI);
		
		~LHE_Pythia_PowerJets();
		
		//////////////////////////////////////////////////////////////////
		
		/*! @brief Generate the next event and analyze
		 * 
		 *  \returns Returns the status of the generator.
		*/
		Status Next() {return Next_internal(true);} // doWork = true
		
		//! @brief Repeat detection (which includes re-generating pileup)
		Status Repeat() {return DoWork();}
		
		void Set_PileupMu(double const pileup_mu);
		
		//! @brief The index of the current event
		size_t EventIndex() const {return nextCount - 1;}
		
		//! @brief The current status of the generator
		Status GetStatus() const {return status;}
		
		//! @brief Get a list of all tracks and towers detected in the event
		std::vector<vec3_t> const& Get_Detected() const {return detected;}
		
		//! @brief Convert the LHE matrix element (parton-level event) into Jet's
		std::vector<Jet> const Get_ME() const 
			{return std::vector<Jet>(detector->ME().cbegin(), detector->ME().cend());}
			
		//! @brief Cluster the detected particles into jets and return
		std::vector<Jet> Cluster_FastJets(bool const subtractPileup = false) const;
		
		//! @brief Return the pileup density (only valid after ClusterFastJets(true))
		double Get_RhoPileup() const;
				
		//~ std::vector<vec4_t> const& Get_Pileup() {return pileup;}		
		
		Settings const& Get_Settings() const {return settings;}
		
		ArrogantDetector const* Get_Detector() const {return detector;}
		
		//! @brief Construct a container of jets with pileup
		ShapedParticleContainer JetContainer(std::vector<ShapedJet> const& jets, 
			real_t const f_pileup = real_t(0)) const;
		
		//! @brief Calculate the power spectrum of the observed final state using the natural resolution
		std::vector<real_t> const& Get_Hl_Obs(size_t const lMax) const;
		
		/*! @brief Calculate the power spectrum of an ensemble of ShapedJet's and pileup
		 * 
		 *  The pileup shape is read upon initialization.
		*/ 
		std::vector<real_t> Get_Hl_Jet(size_t const lMax, 
			std::vector<ShapedJet> const& jets, real_t const f_pileup = real_t(0)) const;
		
		/*! @brief Calculate the "hybrid" power spectrum between jets/pileup and the observed final state
		 * 
		 *  See PowerSpectrum::Hl_Hybrid for more details.
		 *  The pileup shape is read upon initialization.
		*/ 
		std::vector<real_t> Get_Hl_Hybrid(size_t const lMax, 
			std::vector<ShapedJet> const& jets, real_t const f_pileup = real_t(0)) const;
		
		/*! @brief Bin all detector into the calorimeter, 
		 *  then write the tower edges and energy to a file.
		 * 
		 *  This is useful for generating LEGO plots
		*/ 
		void WriteAllVisibleAsTowers(std::string const& filePath) const;
		
		//! @brief (tau * h_l^trk + (1-tau) * h_l^twr)
		//~ std::vector<real_t> const& Get_DetectorFilter(size_t const lMax);		
		
		//~ std::vector<vec4_t> const& Get_FinalState() const {return detector->FinalState();}
		/*! @brief Get the observation in the lab frame
		 *  
		 *  TODO: add support for boost.
		*/ 
		//~ DetectorObservation const& Get_Observation() const {return observation;}
		//~ std::vector<vec3_t> const& Get_Tracks() const {return detector->Tracks();}
		//~ std::vector<vec3_t> const& Get_Towers() const {return detector->Towers();}
		
		//~ ArrogantDetector::Settings const& Get_DetectorSettings() const {return detector->GetSettings();}
				
		// Using SpectralPower.hpp; used these calls to validate PowerSpectrum
		//~ std::vector<real_t> Get_Hl_Obs_slow(size_t const lMax);
		//~ std::vector<real_t> Get_Hl_Jet_slow(size_t const lMax, std::vector<ShapedJet> const& jets);
		//~ std::vector<real_t> Get_Hl_Hybrid_slow(size_t const lMax, std::vector<ShapedJet> const& jets);		
		
		// These are all easy to calculate using the PowerSpectrum framework
		//~ std::vector<real_t> const& Get_H_showered(size_t const lMax);	
		//~ std::vector<real_t> const& Get_H_det(size_t const lMax);		
		//~ std::vector<real_t> const& Get_H_extensive(size_t const lMax);		
		//~ std::vector<real_t> const& Get_Hl_FinalState(size_t const lMax);
		
		// Give me a vector pointing in an isotropic direction whose 
		// length follows an exponential distribution with mean = 1/lambda
		//~ static vec4_t IsoVec3_Exponential(pqRand::engine& gen, real_t const meanE);
};

GCC_IGNORE_POP

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

#endif
