#ifndef ARROGANT_DETECTOR
#define ARROGANT_DETECTOR

#include "PowerJets.hpp"
#include "kdp/kdpTools.hpp"
#include "Pythia8/Pythia.h"
#include "Pythia8/Event.h"
#include <QtCore/QSettings>

GCC_IGNORE_PUSH(-Wpadded)

/*! @brief A tower object that stores 3-momentum and fractional solid angle (Omega / (4 pi))
 *  
 *  Fields are public, just like the vec3_t it contains (WAAH).
*/
struct Tower
{
	using vec3_t = PowerJets::vec3_t;
	using real_t = PowerJets::real_t;
		
	vec3_t p3; //!< @brief 3-momentum of massless tower
	real_t fOmega; //!< @brief fractional solid angle (Omega / (4 pi))
	
	Tower(vec3_t const& p3_in, real_t const fOmega_in):
		p3(p3_in), fOmega(fOmega_in) {}
	
	operator vec3_t() const {return p3;}
};

/*! @brief A very basic detector for phenomenological studies
 *  
 *  This detector is arrogant because it makes some rather extreme assumptions:
 *   - Any particle with |eta| < etaMax_cal is detected in tracks or towers.
 *   - Any charged particle with |eta| < etaMax_track and |pT| > minPT is detected \em perfectly,
 *     with perfect track subtraction from the calorimeter.
 *   - Because there is no simulation of the magnetic field, 
 *     untracked charged particles (loopers) do not strike the endcap,
 *     propagating in a straight line (as if they were neutral) to the calorimeter.
 *   - There are no material interactions or measurement errors;
 *     all energy and angles are perfect.
 *  
 *  To create a working ArrogantDetector, the derived class must define the functions
 *  NumPhiBins() and PhiWidth() and fill etaBeltEdges in the ctor.
 *  The derived ctor shall also call Init_InDerivedCtor().
 * 
 *  four functions which determine the calorimeter scheme.
 * 
 *  ArrogantDetector's calorimeter uses an equatorial "angle" \a t
 *  (where \f$ \hat{z} \to +t_{max} \f$, \f$ -\hat{z} \to -t_{max} \f$,
 *  and the transverse plane is at \f$ t=0 \f$). The user must define
 *  the equatorial angle via: 
 *   - AbsEquatorialAngle() return \f$ |t| \f$, given a 4-vector.
 *   - ToAbsTheta() maps \f$ |t| \f$ onto the standard equatorial angle theta.
 * 
 *  \f$ |t| \f$ is used to define a bijective map between
 *  detector coordinates and a unique towerID:
 *   - GetTowerID() maps forward detector coordinates to calorimeter \p towerID.
 *   - GetTowerEdges() maps \p towerID to the tower's edges
 * 
 *  The forward (t > 0) and backward (t < 0) halves of the detector are stored separately, 
 *  so the \p towerID scheme only needs to depend on absolute |t|. 
 *  This dramatically simplifies the ID scheme. 
 *  The default \p towerID scheme creates ``square'' towers of 
 *  uniform width in \a t, so the simplest derived class need only 
 *  additionally define the width of towers in \a phi via PhiWidth().
*/

// 1. Read out towers via towerID. 
//     - from edges, calculate geometric center and fractional area
//     - 3-vector and fractional area. 
//     - boost along z-axis, alter z-momentum
//     - from new z-momentum, look up new fractional area
//     - fractional area is identical for every belt!
class ArrogantDetector
{
	/////////////////////////////////////////////////////////////////////
	/*   _____                    _   _                           
	 *  |  ___|   ___    _ __    | | | |  ___    ___   _ __   ___ 
	 *  | |_     / _ \  | '__|   | | | | / __|  / _ \ | '__| / __|
	 *  |  _|   | (_) | | |      | |_| | \__ \ |  __/ | |    \__ \
	 *  |_|      \___/  |_|       \___/  |___/  \___| |_|    |___/
	 * 																						*/
	/////////////////////////////////////////////////////////////////////
	public:
	
		////////////////////////////////////////////////////////////////////////////////////////////////////////
		/*   _____                                   ___        ____   _                                  
		 *  |_   _|  _   _   _ __     ___   ___     ( _ )      / ___| | |   __ _   ___   ___    ___   ___ 
		 *    | |   | | | | | '_ \   / _ \ / __|    / _ \/\   | |     | |  / _` | / __| / __|  / _ \ / __|
		 *    | |   | |_| | | |_) | |  __/ \__ \   | (_>  <   | |___  | | | (_| | \__ \ \__ \ |  __/ \__ \
		 *    |_|    \__, | | .__/   \___| |___/    \___/\/    \____| |_|  \__,_| |___/ |___/  \___| |___/
		 *           |___/  |_|                                                                           		*/
		////////////////////////////////////////////////////////////////////////////////////////////////////////
		 
		typedef typename kdp::Vector3<double> vec3_t; //!< @brief The 3-vector type
		typedef typename kdp::Vector4<double> vec4_t; //!< @brief The 4-vector type
	
		//! @brief The main ArrogantDetector settings, read from a parsed INI file.
		class Settings
		{
			friend class ArrogantDetector;
			
			private:
				static double Read_double(QSettings const& parsedINI, 
					std::string const& detectorName, std::string const& key, 
					double const defaultVal);
					
				// Read in the requested squareWidth, which is the phi width of towers in the central belt, 
				// then round it so that there are an even number of towers in the central belt.
				// Use this integer to define the actual squareWidth
				
				// Hide the ctor from the public
				Settings(QSettings const& parsedINI, std::string const& detectorName);
					
			public:
				static constexpr auto squareWidth_default = "5 deg";
				static constexpr auto etaMax_cal_default = 5.;
				static constexpr auto etaMax_track_default = 2.5;
				static constexpr auto minTrackPT_default = 0.3; // 300 MeV
				// Loopers are not tracked: pT = 0.3 |q| B R (0.3 GeV per Tesla per elementary charge)
				// assume B = 2 T and minR = .5 m, so 300 MeV is a good minimum
							
				double squareWidth; //!< @brief The angular width of calorimeter cells in the central belts
				double etaMax_cal; //!< @brief The maximum pseudorapidity of the calorimeter.
				double etaMax_track; //!< @brief The maximum pseudorapidity of the tracker.
				double minTrackPT; //!< @brief The minimum pT of detectable tracks.
				bool evenTowers; //!< @brief Create an even number of towers in each belt (so each tower at [eta, phi] has an antipodal tower at [-eta, -phi]).
		};
		
		//! @brief A simple struct for communicating tower edges.
		struct Edges {double eta_lower, eta_upper, phi_lower, phi_upper;};
		
		//! @brief Add energy to Edges to facilitate a LEGO plot.
		struct RawTower : public Edges 
		{
			double energy;
			
			RawTower(Edges&& edges, double const energy_in);
				
			void FlipZ();
		};
		
		//////////////////////////////////////////////////////////////////
		/*   _____                          _     _                       
		 *  |  ___|  _   _   _ __     ___  | |_  (_)   ___    _ __    ___ 
		 *  | |_    | | | | | '_ \   / __| | __| | |  / _ \  | '_ \  / __|
		 *  |  _|   | |_| | | | | | | (__  | |_  | | | (_) | | | | | \__ \
		 *  |_|      \__,_| |_| |_|  \___|  \__| |_|  \___/  |_| |_| |___/
		 *                                                                	*/
		//////////////////////////////////////////////////////////////////		
		 
		//! @brief Read the settings from the INI file and set up the detector.
		ArrogantDetector(QSettings const& parsedINI, std::string const& detectorName = "detector");
		
		virtual ~ArrogantDetector() {}
		
		/*! @brief Read "detectorName/type" and return a new detector 
		 *  of that type (which the user is now responsible for cleaning up).
		 * 
		 *  We look for the following keywords in type
		 *  "lep" -> ArrogantDetector_Lepton
		 *  "had" -> ArrogantDetector_Hadron		 *  
		*/ 
		static ArrogantDetector* NewDetector(QSettings const& parsedINI, 
			std::string const& detectorName = "detector");
			
		size_t NumTowers() const;
		
		//////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////
		
		void Clear(); //!< @brief Clear the detector before filling.
		void Finalize(bool const correctMisingE = true); //!< @brief Finalize the detector (e.g., write-out calorimeter) after filling.
		
		//////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////
		
		/*! @brief Fill the detector from the Pythia event.
		 * 
		 *  The ME() vector is filled using Pythia status code +/- 23. 
		 *  Final state particles are split into neutral, charged and invisible and 
		 *  sent to the other operator().
		*/ 
		void operator()(Pythia8::Pythia& event);
		
		/*! @brief Fill the detector from neutral, charged, invisible and pileup.
		 * 
		 *  Charged particles' \em momentum is tracked. They are assumed massless,
		 *  and their excess energy (E - |p|) is deposited into the calorimeter.
		 * 
		 *  Neutral 3-momenta are used to find the struck tower,
		 *  but the energy deposition comes from neutral.x0
		 *  (i.e. the magnitude of neutral 3-momenta does not affect 
		 *  the energy deposited into each tower).
		 * 
		 *  Invisible particles appear only at truth level (FinalState())
		 * 
		 *  Charged pileup tracks are kept separate from regular tracks, 
		 *  under the assumption that their primary vertex is distinguishable.
		*/ 
		void operator()(std::vector<vec4_t> const& neutralVec,
			std::vector<vec4_t> const& chargedVec = std::vector<vec4_t>(), 
			std::vector<vec4_t> const& invisibleVec = std::vector<vec4_t>(), 
			std::vector<vec4_t> const& neutralPileup = std::vector<vec4_t>(), 
			std::vector<vec4_t> const& chargedPileup = std::vector<vec4_t>());
			
		/*! @brief Used to overlap multiple events into a single detection. 
		 *  
		 *  Must be used in conjunction with Clear() and Finalize().
		 *  Useful for adding pileup to a hard scatter.
		 *  
		 *  If /p isPileup = false, then the ME() vector is not altered.
		*/		
		void PartialFill(Pythia8::Pythia& event, bool const isPileup = false);
		
		/*! @brief Used to overlap multiple events into a single detection. 
		 * 
		 *  Must be used in conjunction with Clear() and Finalize().
		 *  Useful for adding pileup to a hard scatter.
		*/
		void PartialFill(std::vector<vec4_t> const& neutralVec,
			std::vector<vec4_t> const& chargedVec = std::vector<vec4_t>(), 
			std::vector<vec4_t> const& invisibleVec = std::vector<vec4_t>(), 
			std::vector<vec4_t> const& neutralPileup = std::vector<vec4_t>(), 
			std::vector<vec4_t> const& chargedPileup = std::vector<vec4_t>());		
		
		//////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////
				
		//! @brief Get the matrix element.
		inline std::vector<Pythia8::Particle> const& ME() const {return me;}
		
		//! @brief Get the truth-level final-state particles (including neutrinos).
		inline std::vector<vec4_t> const& FinalState() const {return finalState;}
		
		//! @brief Get all charged particles passing eta and pT cuts.
		inline std::vector<vec3_t>const& Tracks() const {return tracks;}
		
		//! @brief Get all charged particles from PU vertices passing eta and pT cuts.
		inline std::vector<vec3_t>const& PileupTracks() const {return tracks_PU;}
		
		/*! @brief Get towers from the calorimeter
		 * 
		 *  \warning Depending on how AddMissingE is configured, 
		 *  the last tower may be the missing energy.
		*/
		inline std::vector<Tower> const& Towers() const {return towers;}
		
		inline Settings const& GetSettings() const {return settings;}
		
		//! @brief Return the edges of every tower, with fractional area stored in energy.
		std::vector<RawTower> GetAllTowers() const;
		
		//! @brief Return all visible energy in calorimeter towers (via edges)
		std::vector<RawTower> AllVisible_InTowers();		
		
		//////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////
		//! @defgroup etaTheta    Convert between eta(pseudorapidity), theta(equatorial angle), and tan(theta)
		
		//! @ingroup etaTheta
		static double Eta_to_Theta(double const eta) {return std::atan(std::sinh(eta));}
		
		//! @ingroup etaTheta
		static double Eta_to_TanTheta(double const eta) {return std::sinh(eta);}
		
		//! @ingroup etaTheta	
		static double TanTheta_to_Eta(double const tanTheta) {return std::asinh(tanTheta);}
		
		//! @ingroup etaTheta
		static double Theta_to_Eta(double const theta) {return std::asinh(std::tan(theta));}
		
	protected:
		//////////////////////////////////////////////////////////////////
		/*   _____             ____                  _                
		 *  |_   _|   ___     |  _ \    ___   _ __  (_) __   __   ___ 
		 *    | |    / _ \    | | | |  / _ \ | '__| | | \ \ / /  / _ \
		 *    | |   | (_) |   | |_| | |  __/ | |    | |  \ V /  |  __/
		 *    |_|    \___/    |____/   \___| |_|    |_|   \_/    \___|    */
		//////////////////////////////////////////////////////////////////                                                           
		
		/*! @brief Each tower is identified with a unique ID.
		 * 
		 *  This should be an unsigned integer. We choose a large type so that
		 *  maxPhiBinsPerBelt can be fixed to a large number, 
		 *  leaving \em plenty of IDs for any calorimeter scheme.
		*/
		typedef uint64_t towerID_t;
				
		/*! @brief The maximum number of phi bins in each equatorial belt.
		 * 
		 *  To balance the number of belts with the number of phiBins, 
		 *  we use the square root of the number of unique IDs.
		*/ 
		static constexpr towerID_t maxTowersPerBelt = 
			(towerID_t(1) << (std::numeric_limits<towerID_t>::digits / 2));
			
		//! @brief A class to ensure that deltaPhi is is correctly mapped to the unit circle
		class PhiSpec
		{
			private:
				double deltaPhi;
				towerID_t numTowers;
				
			public:
				PhiSpec(double const deltaPhi_target, bool forceEven = false);
			
				double DeltaPhi() const {return deltaPhi;}
				towerID_t NumTowers() const {return numTowers;}
				
				operator double() const {return deltaPhi;}
		};
		
		//////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////		
			
		ArrogantDetector::Settings settings; //<! @brief Settings read by the ctor.
			
		/*! @brief The eta boundaries of calorimeter belts. Filled by the derived ctor.
		 * 
		 *  \warning The derived ctor \em SHALL fill only the lower edges.
		*/
		std::vector<double> beltEdges_eta;
		
		/*! @brief The width/number of the phi bins in each equatorial calorimeter belt.
		 * 
		 *  @param beltIndex The index of the equatorial calorimeter belt
		*/
		virtual PhiSpec const& DeltaPhi(towerID_t const beltIndex) const = 0;
		
		/*! @brief Deal with missing energy after filling tracks and towers
		 *  (and adding them all to \p visibleP3).
		 * 
		 *  By default, missing energy is opposite all observed energy.
		*/ 
		virtual void AddMissingE();
		
		/*! @brief Initalize the calorimeter from beltEdges_eta and DeltaPhi().
		 * 
		 *  This must be called AFTER the derived ctor has filled beltEdges_eta.
		*/
		void Init_inDerivedCTOR();
		
		//////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////
		// Derived class has access to these so it can redefine missingE scheme
		
		// The particles for the last detected event
		std::vector<Pythia8::Particle> me; // matrix element, read from Pythia file
		std::vector<vec4_t> finalState; // all final state particles
		std::vector<vec3_t> tracks, tracks_PU; // visible charged particles
		std::vector<Tower> towers; // calorimeter towers
		
		// Running sums while filling the detector
		vec3_t visibleP3; // 3-momentum sum of reconstructed (massless) particles
		vec4_t invisibleP4; // 4-momentum sum of invisible particles
		double visibleE, pileupE; // visible energy & charged pileup E = \sum |p|
		// keep visibleE and visibleP3 separate because charged tracks are seen via massless momentum, 
		// which is slightly smaller than track energy (i.e. a sum of 4-vectors) 
			
	private:
		//////////////////////////////////////////////////////////////////
		/*   _   _   _       _       _          
		 *  | | | | (_)   __| |   __| |   ___   _ __  
		 *  | |_| | | |  / _` |  / _` |  / _ \ | '_ \ 
		 *  |  _  | | | | (_| | | (_| | |  __/ | | | |
		 *  |_| |_| |_|  \__,_|  \__,_|  \___| |_| |_|
		 * 
		 *   ___                       _                                     _             _     _                 
		 *  |_ _|  _ __ ___    _ __   | |   ___   _ __ ___     ___   _ __   | |_    __ _  | |_  (_)   ___    _ __  
		 *   | |  | '_ ` _ \  | '_ \  | |  / _ \ | '_ ` _ \   / _ \ | '_ \  | __|  / _` | | __| | |  / _ \  | '_ \ 
		 *   | |  | | | | | | | |_) | | | |  __/ | | | | | | |  __/ | | | | | |_  | (_| | | |_  | | | (_) | | | | |
		 *  |___| |_| |_| |_| | .__/  |_|  \___| |_| |_| |_|  \___| |_| |_|  \__|  \__,_|  \__| |_|  \___/  |_| |_|
		 *                    |_|                                                                                   */
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		
		/*! @brief The towerID, based upon a particle's coordinates in the forward detector.
		 * 
		 *  The calorimeter is segmented into equatorial belts formed from towers sharing 
		 *  the same boundaries in equatorial angle (i.e. latitude), 
		 *  but different boundaries in phi (i.e. longitude).
		 *  Each tower has a unique ID that is a combination of it's 
		 * 
		 *  	- beltIndex: the index of its equatorial belt (index 0 starts at eta = 0)
		 * 	- phiIndex: the index of the tower in the belt (index 0 starts at phi = -Pi)
		 * 
		 *  We then calculate
		 * 
		 *  	towerID = beltIndex * maxTowersPerBelt + phiIndex.
		 * 
		 *  Because maxTowersPerBelt is often much larger than the
		 *  number of towers in the actual belts, many towerIDs do not map 
		 *  to an actual tower. However, by using a large, static choice for 
		 *  maxTowersPerBelt, the math can be vastly simplified.
		 *  The validity of a towerID is checked by any function which
		 *  uses phiIndex to look something up. 
		 * 
		 *  For simplicity, the same IDs map both the forward and back detector
		 *  (since beltIndex = 0 starts at eta = 0), so we keep 
		 *  separate detectors to delineate forward and backward particles.
		 *  Hence, there is no belt straddling eta = 0; the equator
		 *  separates the first belts in the forward and backward detectors.
		 * 
		 *  To facilitate conversion between indices and towerID, we 
		 *  use separate structs (TowerID and TowerIndices) to store each, 
		 *  then add operators to allow implicit conversion like:
		 *  	
		 *  	towerID_t <--> TowerID <--> TowerIndices
		 * 
		 *  This scheme allows TowerID to be used directly as an index, 
		 *  and also in a std::set or std::map by harnessing 
		 *  operator< for towerID_t. What is NOT ALLOWED is a direct 
		 *  conversion from TowerIndices to towerID_t, which is 
		 *  somewhat ambiguous (are we getting the towerID or the beltIndex?).
		 *  Hence, TowerIndices cannot be accidentally used as an index.
		 */  
		class TowerID;
		
		//! @brief Each tower is uniquely identified via two indices
		struct TowerIndices
		{
			towerID_t beltIndex; // the index of its equatorial belt (index 0 starts at eta = 0)
			towerID_t phiIndex; // the index of the tower within the belt (via phi)
			
			TowerIndices(TowerID towerID); // Break a towerID into its component indices
			TowerIndices(towerID_t const beltIndex_in, towerID_t const phiIndex_in):
				beltIndex(beltIndex_in), phiIndex(phiIndex_in) {}
			
			operator TowerID() const; // Convert indices to their towerID
		};
		
		// We cannot extend towerID_t, because it is not a class.
		// However, the towerID_t operator allows it to be used like a towerID_t.
		struct TowerID
		{
			towerID_t id;
			
			TowerID(towerID_t const id_in):id(id_in) {}
			
			//~ operator TowerIndices() const {return TowerIndices(id);} // compiler can see this ctor, get's confused if there's a redundant operator
			operator towerID_t() const {return id;}
		};
	
		/*! @brief The calorimeter is a map between TowerID and energy.
		 *  This scheme saves a lot of memory and time compared to the naive scheme,
		 *  where you mimic the calorimeter with an enormous 2D histogram of binned energy 
		 *  (which is usually mostly zeroes). In fact, the mapping scheme 
		 *  allows one to use an extremely fine-grained calorimeter
		 *  with no loss of performance (since the number of hits is 
		 *  limited by the event multiplicity).
		 *  The mapping scheme is only worse than the explicit histogram 
		 *  when nearly ever calorimeter cell is filled, 
		 *  which is probably unlikely in the intended use case of this class.
		*/ 
		typedef std::map<TowerID, double> cal_t;
		
		// To simplify the ID scheme (i.e., it only cares about |eta|)
		// we keep separate forward/backward calorimeters 
		cal_t foreCal, backCal;		
		
		// The derived ctor fills beltaLowerEdges_eta, but we look-up towers
		// via their tanTheta, which takes less math to calculate.
		// We therefore map eta -> tanTheta in Init_inDerivedCTOR().
		std::vector<double> beltEdges_tanTheta;
		
		// Boundary values based on settings.etaMax_cal and settings.etaMax_track
		double tanThetaMax_cal; //!< @brief The maximum equatorial angle for calorimeter coverage.
		double tanThetaMax_track; //!< @brief The maximum equatorial angle for tracker coverage.
		
		towerID_t numBelts; //!< @brief The number of equatorial belts in the calorimeter.
		towerID_t tooBigID; //!< @brief One past the largest valid \p towerID.
		
		/*! @brief The prototype geometric center for each belt
		 * 
		 *  Since each belt shares the same equatorial boundaries, 
		 *  each tower's geometric center is at the same eta; 
		 *  the only difference is its phi location.
		 *  To convert towerPrototype to a given phi location, 
		 *  its transverse components need only be multiplied by [cos(phi), sin(phi)].
		*/ 
		std::vector<vec3_t> towerPrototype;
		
		//!< @brief Every tower in a belt shares the same fractional area fA = Omega / (4 Pi)
		std::vector<double> fractionalArea; 
		
		//!< @brief Used to ensure that Finalize() is called once per Clear().
		bool finalized;
		
		//////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////
		
		//! @brief Because we keep separate forward/backward detectors, 
		//! we need to distinguish forward and backward particles.
		static bool IsForward(vec3_t const& p3) {return (p3.x3 >= 0.);}
				
		/*! @brief Fill the calorimeter int towers. 
		 * 
		 *  Since TowerID maps to |eta|, we must flip the sign of p.x3 (z-coord) 
		 *  when writing the backward cal. The user must let us know.
		*/ 
		void WriteCal(cal_t const& cal, bool const flipZ);
				
		//////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////
		
		//! @brief Get the equatorial index of the belt of towers
		towerID_t GetBeltIndex_AbsTanTheta(double const absTanTheta) const;
		
		//! @brief Get the phi index of the tower
		towerID_t GetPhiIndex(double phi, towerID_t const beltIndex) const;
		
		//! @brief Given a particle's angular position, obtain its equatorial and phi indices
		TowerIndices GetIndices_AbsTanTheta_Phi(double const absTanTheta, double const phi) const;
			
		//! @brief Given a tower's indices, calculate it's central phi position
		double GetCentralPhi(TowerIndices const& indices) const;
		
		//! @brief Given a tower's ID, calculate it's geometric center
		Tower GetTowerCenter(TowerIndices const& indices) const;
		
		//! @brief Return the location of the tower's edges in eta/phi 
		Edges GetTowerEdges(TowerIndices const& indices) const;
};

/*! @brief An ArrogantDetector with constant deltaPhi and towers optimized for squareness.
 * 
 *  Coincidentally, requiring square towers essentially creates
 *  equatorial belts with a nearly constant width in 
 *  pseudorapidity \f$ \eta = {\tt arctanh}\left(\frac{p_L}{p}\right)\f$.
*/ 
class ArrogantDetector_Hadron : public ArrogantDetector
{
	protected:
		PhiSpec deltaPhi;
		
		//! @brief \a phi bins have constant width \p squareWidth
		virtual PhiSpec const& DeltaPhi(__attribute__((unused)) towerID_t const etaIndex) const 
			{return deltaPhi;}
			
	public:
		ArrogantDetector_Hadron(QSettings const& parsedINI, 
			std::string const& detectorName = "detector");
		
		virtual ~ArrogantDetector_Hadron() {}
};

/*! @brief An ArrogantDetector with equatorial belts of approximately constant deltaTheta, 
 *  optimized so that each tower has approximately the same solid angle.
 * 
 *  The lepton collider uses the equatorial angle \f$ \theta = \arctan\left(\frac{p_L}{p_T}\right)\f$.
 *  The calorimeter is built from towers with approximately constant 
 *  \f$ d\Omega = \cos(\theta)d\theta d\phi \f$.
 *  The calorimeter belts have a constant width \f$ d\theta = {\tt squareWidth} \f$, 
 *  so that \f$ d\phi \approx \frac{d\Omega}{d\theta \cos(\theta)} \approx \frac{\tt squareWidth}{{\cos(\theta)}}\f$, 
 *  where \f$ d\phi \f$ is rounded to the closest integer number of 
 *  \a phi bins per belt.
*/ 
class ArrogantDetector_EqualArea : public ArrogantDetector
{
	protected:
		//! @brief Return the phi width of towers in a given belt, along with their count
		virtual PhiSpec const& DeltaPhi(towerID_t const beltsIndex) const;
		
	private:
		// Each theta belt has a different deltaPhi, to maintain dOmega.
		// To implement the bijective towerID mapping quickly, we store it.
		std::vector<PhiSpec> belt_deltaPhi;
		
		PhiSpec const& EmplaceDeltaPhi(double const deltaPhi_target);
		
	public:
		ArrogantDetector_EqualArea(QSettings const& parsedINI, 
			std::string const& detectorName = "detector");
		
		virtual ~ArrogantDetector_EqualArea() {}
};

typedef ArrogantDetector_EqualArea ArrogantDetector_Lepton;

GCC_IGNORE_POP

#endif
