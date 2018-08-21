#ifndef ARROGANT_DETECTOR
#define ARROGANT_DETECTOR

// Copyright (C) 2018 by Keith Pedersen (Keith.David.Pedersen@gmail.com)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "PowerJets.hpp"
#include "PowerSpectrum.hpp"
#include "kdp/kdpTools.hpp"
#include "Pythia8/Pythia.h"
#include "Pythia8/Event.h"
#include <QtCore/QSettings>

#include "kdp/kdpSettings.hpp"

/*! @file ArrogantDetector.hpp
 *  @brief Defines an abstract detector classes and two implementations
 *  @author Copyright (C) 2018 Keith Pedersen (Keith.David.Pedersen@gmail.com, https://wwww.hepguy.com)
*/ 

GCC_IGNORE_PUSH(-Wpadded)
 
////////////////////////////////////////////////////////////////////////

/*! @brief A very basic particle detector for phenomenological studies
 *  
 *  This detector class is designed to replicate the very basic 
 *  functionality of an LHC-like particle detector (where the 
 *  longitudinal direction is defined to be parallel to the colliding beams).
 *  Such a detector has two major components:
 * 
 *   - A "tracker" to detect the passage of charged particles.
 *     - Charged particles ionize the detection medium, 
 *     leaving highly-localized (sub-mm) blips that can be 
 *     connected into a track with extremely good angular resolution.
 *     A longitudinal magnetic field (parallel to the beam line)
 *     deflects the charged particles, with each track's 
 *     radius of curvature being proportional to its momentum.
 * 
 *   - A "calorimeter" to detect particle energy. 
 *      - Calorimeters are nearly hermitic sensors (two holes for the beam line)
 *     that measure the kinetic energy of moving particles.
 *     They are segmented into square cells, and in a crude sense, 
 *     can only determine how much energy is deposited into each cell 
 *     (not the location within the cell where this energy was deposited).
 *     This gives the calorimeter much poorer angular resolution than the tracker.
 * 
 *      - The energy deposited in each cell is called a "tower" 
 *     (because if you unroll the surface of the calorimeter onto a plane, 
 *     and show the energy in each cell with a 2D histogram, 
 *     each active cell looks like a tower or skyscraper). 
 *     Both charged and neutral particles deposit their energy into towers,
 *     but charged particles are also seen in the tracker.
 *     Their tracks can be extrapolated to the tower they struck, 
 *     and their energy subtracted (via their well-measured momentum).
 *     This leaves a tower of neutral energy. *     
 * 
 *  In our initial investigations of the angular power spectrum of QCD jets, 
 *  it is important to understand the implications of the different 
 *  angular resolutions provided by tracks and towers.
 *  We therefore simulate a detector with perfect energy resolution, 
 *  and only the most bare-bones approximation of angular resolution.
 * 
 *   - Any particle with pseudorapidity \f$ |\eta| < \eta_{\max}^{\rm cal} \f$
 *     is detected.
 * 
 *   - Any charged particle with \f$ |\eta| < \eta_{\max}^{\rm trk} \f$ and 
 *     transverse momentum \f$ |p_T| > p_T^{\min,\rm trk} \f$ is detected \em perfectly,
 *     with perfect track subtraction from the calorimeter.
 * 
 *   - We assume there is a magnetic field (to measure track momentum), 
 *     but do not simulate track deflection. Therefore, 
 *     untracked charged particles (loopers) do not strike the endcap,
 *     instead propagating in a straight line (as if they were neutral) to the calorimeter.
 * 
 *   - There are no material interactions or measurement errors;
 *     all energy and angles are perfect.
 *  
 *  The calorimeter is composed of belts of towers, 
 *  where each tower in a given belt has the same pseudorapidity boundaries
 *  and the same width in phi (DeltaPhi()). See TowerID for more information.
 *  
 *  The calorimeter is configured via the Settings read from an INI file
 *  (see \ref ArrogantDetector::Settings).
 *   
 *  To create a working ArrogantDetector, the derived class must define
 *  the calorimeter grid by defining PhiWidth() and, in its derived ctor, 
 *  fill beltEdges_eta with the lower edges (in pseudorapidity eta) 
 *  of every calorimeter belt, as well as the maximum eta of the final belt
 *  (which shall be <= settings.etaMax_cal).
 *  The derived ctor shall then call Init_InDerivedCtor().
*/
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
		 
		using vec3_t = kdp::Vector3<double>; //!< @brief The 3-vector type
		using vec4_t = kdp::Vector4<double>; //!< @brief The 4-vector type
		using DetectorObservation = PowerSpectrum::DetectorObservation;
		
		static constexpr char const* detectorName_default = "Detector";
	
		/*! @brief The main ArrogantDetector settings, read from a parsed INI file
		 *  from section "[detectorName]"
		 * 
		 *  Each parameter is defined using <tt> Param<T>(key, default value) </tt>
		*/ 
		class Settings : public kdp::Settings_Base
		{
			friend class ArrogantDetector;
			
			private:
				//! @brief Read in all settings, rounding squareWidth so that 
				//! there are an integer number of phi bins in the central belt
				Settings(QSettings const& parsedINI, std::string const& detectorName);
					
			public:
				~Settings() {}
				
				/*! @brief The phi width of calorimeter cells in the central belts
				 * 
				 *  Angles are read using kdp::ReadAngle (support for "deg" and "rad" suffix).
				*/ 
				Param<double> squareWidth = Param<double>("squareWidth", "5 deg");
				
				//! @brief The maximum pseudorapidity of the calorimeter.
				Param<double> etaMax_cal = Param<double>("etaMax_cal", 5.);
				
				//! @brief The maximum pseudorapidity of the tracker.
				Param<double> etaMax_track = Param<double>("etaMax_track", 2.5);
				
				/*! @brief The minimum \f$ p_T \f$ of charged particles which can be tracked
				 *  
				 *  Charged particles are deflected by the magnetic field, 
				 *  creating helical tracks whose radius of curvature is proportional 
				 *  to their transverse momentum. When \f$ p_T \f$ is too low, 
				 *  the radius of curvature is so small that the charged particles
				 *  don't traverse enough layers of tracker to be detected. 
				 *  These "loopers" are not tracked. 
				 *  
				 *  The link between momentum and radius is  \f$ p_T = 0.3 |q| B R \f$ 
				 *  (0.3 GeV per Tesla per elementary charge per meter in helical radius)
				 *  Assuming B = 2 T and minR = .5 m, we get 300 MeV
				*/  
				Param<double> minTrackPT = Param<double>("minTrackPT", 0.3);
				
				//! @brief Force an even number of towers in each belt 
				//!  (so each tower at [eta, phi] has an antipodal tower at [-eta, -phi]).
				Param<bool> evenTowers = Param<bool>("evenTowers", false);
		};
		
		//! @brief A simple struct for communicating tower edges (lower < upper by definition)
		struct Edges {double eta_lower, eta_upper, phi_lower, phi_upper;};
		
		//! @brief Add energy to Edges to facilitate a LEGO plot.
		struct RawTower : public Edges 
		{
			double energy;
			
			RawTower(Edges&& edges, double const energy_in);
			
			//! @brief Flip z-location, which makes eta values their additive inverse.
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
		 
		/*! @brief Read the settings from the INI file and set up the detector.
		 * 
		 *  \param detectorName
		 *  The name of the section (i.e. [section]) in the INI file which defines the detector's parameters
		*/ 
		ArrogantDetector(QSettings const& parsedINI, std::string const& detectorName = detectorName_default);
		
		virtual ~ArrogantDetector() {}
		
		/*! @brief Read the settings in the INI file return a new detector
		 * 
		 *  The "type" of detector is specified by the key "type"
		 * 
		 *  We look for the following values in type
		 *  "lep" -> ArrogantDetector_Lepton
		 *  "had" -> ArrogantDetector_Hadron 
		 * 
		 *  \param detectorName
		 *  The name of the section (i.e. [section]) in the INI file which defines the detector's parameters
		*/ 
		static ArrogantDetector* NewDetector(QSettings const& parsedINI, 
			std::string const& detectorName = detectorName_default);
			
		size_t NumTowers() const; //!< @brief The total number of towers in the calorimeter grid
		
		//////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////
		
		void Clear(); //!< @brief Clear the detector before filling.
		void Finalize(bool const correctMisingE = true); //!< @brief Finalize the detector (e.g., write-out calorimeter) after filling.
		
		//////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////
		
		/*! @brief Fill the detector from the Pythia event.
		 * 
		 *  The ME() (matrix element) vector is filled using Pythia status code +/- 23. 
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
		inline std::vector<vec3_t> const& Towers() const {return towers;}
		
		/*! @brief Apply a longitudinal boost to all detected particles,
		 *  then return a normalized DetectorObservation
		 * 
		 *  The surfaceFraction returned for each tower is determined by 
		 *  asking where the tower WOULD HAVE fallen had the event 
		 *  been detected in the boosted frame.
		*/ 
		DetectorObservation GetObservation(double const beta_longitudenalBoost = 0.) const;
		
		inline Settings const& GetSettings() const {return settings;}
		
		/*! @brief Return the edges of every tower in the calorimeter grid, 
		 *  with their surfaceFraction stored in RawTower::energy.
		*/ 
		std::vector<RawTower> GetAllTowers() const;
		
		//! @brief Bin all visible energy in calorimeter towers; useful for LEGO plots.
		std::vector<RawTower> AllVisible_InTowers() const;	
		
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
		
		//! @brief The absolute tangent of the equatorial angle
		static double AbsTanTheta(vec3_t const& p3) {return std::fabs(p3.x3)/p3.T().Mag();}
		
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
				
		/*! @brief The maximum number of phi bins in each belt of towers.
		 * 
		 *  To balance the number of belts with the number of phiBins, 
		 *  we use the square root of the number of unique IDs.
		*/ 
		static constexpr towerID_t maxTowersPerBelt = 
			(towerID_t(1) << (std::numeric_limits<towerID_t>::digits / 2));
			
		//! @brief A class to ensure that deltaPhi is is correctly mapped to
		//! an integer number of phi bins circumscribing the circle
		class PhiSpec
		{
			private:
				double deltaPhi;
				towerID_t numTowers;
				
			public:
				/*! @brief Takes a deltaPhi_target and rounds it to perfectly fill
				 *  360 degrees with an integer number of towers. 
				 * 
				 *  deltaPhi < 2 Pi (i.e., there is always at least one tower)
				 * 
				 *  if forceEven == true, an even number of towers are forced
				*/ 
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
		 *  \warning The derived ctor \em SHALL fill all edges.
		*/
		std::vector<double> beltEdges_eta;
		
		/*! @brief The width/number of the phi bins in each calorimeter belt.
		 * 
		 *  @param beltIndex The index of the calorimeter belt
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
		 *  \warning The derived ctor \em SHALL fill call this function 
		 *  \em AFTER filling beltEdges_eta.
		*/
		void Init_inDerivedCTOR();
		
		//////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////
		// Derived class has access to these so it can redefine missingE scheme
		
		std::vector<Pythia8::Particle> me; //!< @brief The matrix element, read from Pythia file
		std::vector<vec4_t> finalState; //!< @brief All final state particles
		std::vector<vec3_t> tracks; //!< @brief Visible charged particles
		std::vector<vec3_t> tracks_PU; //!< @brief Visible charged pileup
		std::vector<vec3_t> towers; //!< @brief Calorimeter towers
		std::vector<double> towerAreas; //!< @brief The surface tower fractions
		
		// Running sums while filling the detector
		vec3_t visibleP3; //!< @brief 3-momentum sum of reconstructed (massless) particles
		vec4_t invisibleP4; //!< @brief 4-momentum sum of invisible particles (truth-level info)
		double visibleE; //!< @brief Visible energy
		double pileupE; //!< @brief Charged pileup \f$ E = \sum |p| \f$
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
		
		
		/*! @brief Each tower in the calorimeter has a unique ID, based upon its location
		 * 
		 *  The calorimeter is segmented into belts formed from towers sharing 
		 *  the same boundaries in equatorial angle (i.e. latitude), 
		 *  but different boundaries in phi (i.e. longitude).
		 *  Each tower has a unique ID that is a combination of it's 
		 * 
		 *   - \c beltIndex: the index of its calorimeter belt (index 0 starts at eta = 0)
		 * 
		 *   - \c phiIndex: the index of the tower in the belt (index 0 starts at phi = -Pi)
		 * 
		 *  We then calculate
		 * 
		 *  	ID = beltIndex * maxTowersPerBelt + phiIndex.
		 * 
		 *  Because \ref maxTowersPerBelt is often much larger than the
		 *  number of towers in the actual belts, many TowerID's do not map 
		 *  to an actual tower. Nonetheless, using a large, static choice for 
		 *  \ref maxTowersPerBelt vastly simplifies the math.
		 *  The validity TowerID's are checked by many functions which use them.
		 * 
		 *  For simplicity, the same TowerID's map both 
		 *  the forward (+z) and backward (-z) calorimeters
		 *  (since \c beltIndex = 0 starts at \f$ \eta = 0 \f$), so we keep 
		 *  separate calorimeters to delineate forward and backward particles.
		 *  Hence, there is no belt centered at \f$ \eta = 0 \f$); 
		 *  the equator separates the forward and backward detectors.
		 * 
		 *  To facilitate conversion between indices and TowerID, 
		 *  we use separate structs (TowerID and TowerIndices) to store each, 
		 *  then add operators to allow implicit conversion like:
		 *  	
		 *  	towerID_t <--> TowerID <--> TowerIndices
		 * 
		 *  This scheme allows TowerID to be used directly as an integer index,
		 *  and also in a std::set or std::map, by coverting it to \ref towerID_t
		 *  (e.g., to harness operator< for \ref towerID_t during set inerstion).
		 *  This scheme \em forbids conversion from TowerIndices to \ref towerID_t in one step, 
		 *  so that TowerIndices cannot be accidentally used as an index.
		 */  
		struct TowerID
		{
			// We cannot extend towerID_t, because it is not a class.
			// However, the towerID_t operator allows it to be used like a towerID_t.
			towerID_t ID;
			
			TowerID(towerID_t const id):ID(id) {}
			
			operator towerID_t() const {return ID;}
		};
		
		//! @brief Each tower is uniquely identified via two indices
		struct TowerIndices
		{
			towerID_t beltIndex; //!< @brief The index of the belt (index 0 starts at eta = 0)
			towerID_t phiIndex; //!< @brief The index of the tower within the belt (via phi)
			
			TowerIndices(TowerID towerID); //!< @brief Break a towerID into its component indices
			
			TowerIndices(towerID_t const beltIndex_in, towerID_t const phiIndex_in):
				beltIndex(beltIndex_in), phiIndex(phiIndex_in) {}
			
			operator TowerID() const; //!< @brief Convert indices to their towerID
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
		
		towerID_t numBelts; //!< @brief The number of belts in the calorimeter.
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
		
		//!< @brief Every tower in a belt shares the same surfaceFraction fA = Omega / (4 Pi)
		std::vector<double> surfaceFraction;
		
		kdp::WelfordEstimate<double> meanArea;
		bool equalArea;
		
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
		
		//! @brief Get the index of the belt of towers
		towerID_t GetBeltIndex_AbsTanTheta(double const absTanTheta) const;
		
		//! @brief Get the phi index of the tower
		towerID_t GetPhiIndex(double phi, towerID_t const beltIndex) const;
		
		//! @brief Given a particle's angular position, obtain its indices
		TowerIndices GetIndices_AbsTanTheta_Phi(double const absTanTheta, double const phi) const;
			
		//! @brief Given a tower's indices, calculate it's central phi position
		double GetCentralPhi(TowerIndices const& indices) const;
		
		//! @brief Given a tower's ID, calculate it's geometric center
		vec3_t GetTowerCenter(TowerIndices const& indices) const;
		
		//! @brief Return the location of the tower's edges in eta/phi 
		Edges GetTowerEdges(TowerIndices const& indices) const;
};

/*! @brief An ArrogantDetector with constant deltaPhi and towers optimized for "squareness".
 * 
 *  Coincidentally, requiring square towers essentially creates
 *  belts with a nearly constant width in pseudorapidity 
 *  \f$ \eta = {\tt arctanh}\left(\frac{p_L}{p}\right)\f$, 
 *  just like detectors at the LHC.
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

/*! @brief An ArrogantDetector with belts of approximately constant deltaTheta, 
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

//~ /*! @brief A tower object that stores 3-momentum and fractional solid angle (Omega / (4 pi))
 //~ *  
 //~ *  Fields are public, just like the vec3_t it contains (WAAH).
//~ */
//~ struct Tower
//~ {
	//~ using vec3_t = PowerJets::vec3_t;
	//~ using real_t = PowerJets::real_t;
		
	//~ vec3_t p3; //!< @brief 3-momentum of massless tower
	//~ real_t fOmega; //!< @brief fractional solid angle (Omega / (4 pi))
	
	//~ Tower(vec3_t const& p3_in, real_t const fOmega_in):
		//~ p3(p3_in), fOmega(fOmega_in) {}
	
	//~ operator vec3_t() const {return p3;}
//~ };

#endif
