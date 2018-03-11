#ifndef ARROGANT_DETECTOR
#define ARROGANT_DETECTOR

#include "kdp/kdpTools.hpp"
#include "SpectralPower.hpp" // Need to define vector types
#include "Pythia8/Pythia.h"
#include "Pythia8/Event.h"
#include <QtCore/QSettings>

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
 *  To create a working ArrogantDetector, the user must define
 *  four functions which determine the calorimeter scheme.
 * 
 *  ArrogantDetector's calorimeter uses a polar "angle" \a t
 *  (where \f$ \hat{z} \to +t_{max} \f$, \f$ -\hat{z} \to -t_{max} \f$,
 *  and the transverse plane is at \f$ t=0 \f$). The user must define
 *  the poler angle via: 
 *   - AbsPolarAngle() return \f$ |t| \f$, given a 4-vector.
 *   - ToAbsTheta() maps \f$ |t| \f$ onto the standard polar angle.
 * 
 *  \f$ |t| \f$ is used to define a bijective map between
 *  detector coordinates and a unique towerID:
 *   - GetTowerID() maps forward detector coordinates to calorimeter \p towerID.
 *   - GetTowerCenter() maps \p towerID to the tower's unit position vector in the forward detector.
 * 
 *  The forward (t > 0) and backward (t < 0) halves of the detector are stored separately, 
 *  so the \p towerID scheme only needs to depend on absolute |t|. 
 *  This dramatically simplifies the ID scheme. 
 *  The default \p towerID scheme assumes calorimeter towers of 
 *  uniform square width in \a t, so the simplest derived class need only 
 *  additionally define the width of towers in \a phi via PhiWidth().
*/
class ArrogantDetector
{
	public:
		/*! @brief The towerID scheme should use an unsigned integer.
		 *  
		 *  @note 32 bits can index over 4 billion calorimeter towers; this should be plenty.
		*/
		typedef uint32_t towerID_t;
		typedef typename kdp::Vector3<double> vec3_t; //!< @brief The 3-vector type
		typedef typename kdp::Vector4<double> vec4_t; //!< @brief The 4-vector type		
		//~ typedef typename SpectralPower::PhatF PhatF; //!< @brief The struct for pHat and f
	
		//! @brief The main ArrogantDetector settings, read from a parsed INI file.
		struct Settings
		{
			private:
				static towerID_t Read_numPhiBins_central(QSettings const& parsedINI, 
					std::string const& detectorName, char const* const defaultValue);
			
				static double Read_double(QSettings const& parsedINI, 
					std::string const& detectorName, std::string const& key, 
					double const defaultVal);
				
			public:
				/*! @brief The number of phi bins in the central bands.
				 *  
				 *  Set by the ctor based upon squareWidth.
				*/  
				towerID_t const numPhiBins_centralBand; 
				double const squareWidth; //!< @brief The angular width of calorimeter cells near the central band.
				double const etaMax_cal; //!< @brief The maximum pseudorapidity of the calorimeter.
				double const etaMax_track; //!< @brief The maximum pseudorapidity of the tracker.
				double const minTrackPT; //!< @brief The minimum pT of detectable tracks.
							
				Settings(QSettings const& parsedINI, std::string const& detectorName):
					numPhiBins_centralBand(Read_numPhiBins_central(parsedINI, detectorName, "5 deg")), // 0.087 rad
					squareWidth((2.*M_PI)/double(numPhiBins_centralBand)),
					etaMax_cal(Read_double(parsedINI, detectorName, "etaMax_cal", 5.)),
					etaMax_track(Read_double(parsedINI, detectorName, "etaMax_track", 2.5)),
					// Loopers are not tracked: pT = 0.3 |q| B R (0.3 GeV per Tesla per elementary charge)
					// assume B = 2 T and minR = .5 m, so 300 MeV is a good minimum
					minTrackPT(Read_double(parsedINI, detectorName, "minTrackPT", 0.3))
				{}
		};
			
	protected:
		// Most of these virtual functions should really be static to the derived class,
		// since they neither read nor alter member variables.
		// However, we need to dereference them on an actual object to determine its class
		// and lookup the AbsPolarAngle() definition, so they must be virtual.
		
		//! @brief The absolute polar angle |t| (e.g. |theta|, |eta|).
		virtual double AbsPolarAngle(vec3_t const& particle) const = 0;
		
		//! @brief Convert absolute polar angle |t| to \f$ |\theta| = |\arctan(|p_L|/p_T)| \f$
		virtual double ToAbsTheta(double const absPolarAngle) const = 0;
		
		/*! @brief The towerID, based upon a particle's coordinates in the forward detector.
		 *  
		 *  The default \p towerID scheme assumes that the calorimeter is segmented into 
		 *  azimuthal belts formed from towers sharing the same polar angle \a t. 
		 *  The belts are indexed so that \p polarIndex = 0 corresponds to the most central belt.
		 *  Within each belt, towers are indexed in azimuthal angle \a phi so that 
		 *  \p phiIndex = 0 corresponds to the smallest \a phi (closest to -pi).
		 *  This gives \f$ {\tt towerID} = {\tt polarIndex} \times {\tt maxPhiBins} + {\tt phiIndex} \f$,
		 *  where \p maxPhiBins is the number of towers in the belt with the 
		 *  largest number of towers (in the default case, the most central belt).
		 *  
		 *  The default \p towerID assumes uniforming binning in polar angle \a t
		 *  (using a pitch of \p squareWidth) and, within a single polar belt, 
		 *  uniform binning in azimuthal angle \a phi.
		 *  The width of the \a phi bins within a polar belt is obtained from PhiWidth().
		 * 
		 *  @note In the default scheme, if some belts have less towers than others, 
		 *  then some \p towerID do not correspond to an actual tower.
		 *  However, allocating a uniform number of \p towerID per belt 
		 *  provides the fastest mapping in GetTowerCenter().
		 *  @note Any \p towerID scheme will do, provided it is self-consistent, bijective, and unique.
		*/ 
		virtual towerID_t GetTowerID(double const absPolarAngle, double const phi) const;
		
		/*! @brief Return the unit position vector at the center of this tower.
		 *  
		 *  The default assumes the same binning scheme as the default GetTowerID(), 
		 *  so only positions in the forward half of the detector are returned.
		 */ 
		virtual vec3_t GetTowerCenter(towerID_t const towerID) const;
				
		/*! @brief Return the width of the phi bins in a polar belt.
		 * 
		 *  @param polarIndex The index of the calorimeter belt used in GetTowerID().
		 */
		virtual double PhiWidth(towerID_t const polarIndex) const = 0;
		
		/*! @brief Deal with missing energy after filling tracks and towers
		 *  (and adding them all to \p visibleP3).
		 * 
		 *  By default, missing energy is opposite all observed energy.
		 *  @note If switching to another scheme, be sure to adjust \p visibleE.
		*/ 
		virtual void AddMissingE();		
				
		// Bouundary values
		double polarMax; //!< @brief The maximum possible polar angle.
		double polarMax_cal; //!< @brief The maximum polar angle for calorimeter coverage.
		double polarMax_track; //!< @brief The maximum polar angle for tracker coverage.
		towerID_t tooBigID; //!< @brief One past the largest valid \p towerID.
		
		/*! @brief Initalize boundary values for polar angle \a t and \p towerID 
		 *  (e.g. three \p polarMax and \p tooBigID).
		 *  
		 *  Boundary values are initialized assuming the default, 
		 *  uniform calorimeter scheme of GetTowerID().
		 *  For example, \p etaMax settings are conveted to \p polarAngle, 
		 *  then rounded to the nearest increment of \p squareWidth (i.e. the 
		 *  detector is uniform in polar angle \a t, with grid spacing of squareWidth).
		 * 
		 *  @note This \em must be called by the derived class ctor, so it can use virtual functions.
		*/
		virtual void Init_inDerivedCTOR();
		
		ArrogantDetector::Settings settings; //!< @brief Settings read by the ctor.		
				
	private:
		// The calorimeter is a map between towerID and energy.
		// This scheme saves a lot of memory (and probably time) 
		// versus the naive scheme, where you mimick the calorimeter with an 
		// enormous array of binned energy (which is usually mostly zeroes).
		// The scheme only fails when nearly ever calorimeter cell is filled, 
		// which is probably unlikely in the intended use case of this class.
		typedef typename std::map<towerID_t, double> cal_t;
		
		// To simplify the ID scheme, keep separate forward/backward calorimeters 
		cal_t foreCal, backCal;
		
		// The particles for the last detected event (filled by operator())
		std::vector<Pythia8::Particle> me; // matrix element, read from Pythia file
		std::vector<vec3_t> finalState, // all final state particles
			tracks, towers; // visible particles
		
		// Running sums while filling the detector
		vec3_t visibleP3, invisibleP3; // 3-momentum sum of reconstructed (massless) particles
		double finalStateE, visibleE; // finalStateE => finalState, visiableE => (tracks, towers)
		
		bool clearME; // Should we clear the matrix element vector?
		
		// Because we keep separate forward/backward detectors, 
		// we need to distinguish forward and backward particles.
		static bool IsForward(vec3_t const& p3) {return (p3.x3 >= 0.);}
		
		// To translate \p etaMax to polarAngle \a t using only AbsPolarAngle(),
		// we create a dummy 3-vector. This hack is slow but acceptable since 
		// it is only used during initialization in Init_inDerivedCTOR().
		static vec3_t EtaToVec(double const eta);
				
		// Fill calorimeter into towers, flipping the sign of p.x3 if backward
		void WriteCal(cal_t const& cal, bool const backward);
		
	public:
		//! @brief Read the settings from the INI file and set up the detector.
		ArrogantDetector(QSettings const& parsedINI, 
			std::string const& detectorName = "detector"):
		settings(parsedINI, detectorName), clearME(true) {}
		virtual ~ArrogantDetector() {}
		
		void Clear();
		
		//! @brief Get the matrix element.
		inline std::vector<Pythia8::Particle> const& ME() const {return me;}
		
		//! @brief Get the final state particles (including neutrinos).
		inline std::vector<vec3_t> const& FinalState() const {return finalState;}
		
		//! @brief Get all charged particles passing eta and pT cuts.
		inline std::vector<vec3_t>const& Tracks() const {return tracks;}
		
		//! @brief Get towers from the calorimeter, with missing energy at the back.
		inline std::vector<vec3_t> const& Towers() const {return towers;}
		
		//! @brief Return the location of every tower, weighted by their dOmega / (4 pi)
		std::vector<vec3_t> GetTowerArea() const;
		
		inline Settings const& GetSettings() const {return settings;}
		
		/*! @brief Fill the detector from the Pythia event.
		 * 
		 *  The ME() vector is filled using Pythia status code +/- 23. 
		 *  Final state particles are split into neutral, charged and invisible and 
		 *  sent to the other operator(). 
		 * 
		 *  @param pileup A vector of massless pileup. 
		 * 	The length of the vector is taken to be its energy in GeV. 
		 * 	Pileup is assumed to be neutral (i.e. charged pileup is removeable).
		*/ 
		void operator()(Pythia8::Pythia& event, 
			std::vector<vec4_t> const& pileupVec = std::vector<vec4_t>());
		
		/*! @brief Fill the detector from neutral, charged and invisible particles.
		 * 
		 *  Charged particles' \em momentum is detected. They are assumed massless,
		 *  and their excess energy (E - |p|) is depositied into the calorimeter.
		 *  Neutral 3-momenta are used to find the struck tower,
		 *  but the energy deposition comes from neutral.x0
		 *  (i.e. the magnitude of neutral 3-momenta does not affect 
		 *  the energy deposited into each tower).
		*/ 
		void operator()(std::vector<vec4_t> const& neutralVec,
			std::vector<vec4_t> const& chargedVec = std::vector<vec4_t>(), 
			std::vector<vec4_t> const& invisibleVec = std::vector<vec4_t>(), 
			std::vector<vec4_t> const& pileupVec = std::vector<vec4_t>());
		
		/*! @brief Read "detectorName/type" and return 
		 *  a new detector of that type for which the user becomes reponsigble.
		 * 
		 *  We look for the following keywords in type
		 *  "lep" -> ArrogantDetector_Lepton
		 *  "had" -> ArrogantDetector_Hadron		 *  
		*/ 
		static ArrogantDetector* NewDetector(QSettings const& parsedINI, 
			std::string const& detectorName = "detector");
};

/*! @brief An ArrogantDetector at a lepton collider.
 * 
 *  The lepton collider uses the polar angle \f$ \theta = \arctan\left(\frac{p_L}{p_T}\right)\f$.
 *  The calorimeter is built from towers with approximately constant 
 *  \f$ d\Omega = \cos(\theta)d\theta d\phi \f$.
 *  The calorimeter belts have a constant width \f$ d\theta = {\tt squareWidth} \f$, 
 *  so that \f$ d\phi \approx \frac{d\Omega}{d\theta \cos(\theta)} \approx \frac{\tt squareWidth}{{\cos(\theta)}}\f$, 
 *  where \f$ d\phi \f$ is rounded to the closest integer number of 
 *  \a phi bins per belt.
*/ 
class ArrogantDetector_Lepton : public ArrogantDetector
{
	protected:
		//! @brief \f$ |\theta| = \arctan\left(\frac{|p_L|}{p_T}\right) \f$.
		virtual double AbsPolarAngle(vec3_t const& particle) const;
		
		//! @brief polar angle is already theta.
		virtual double ToAbsTheta(double const absTheta) const {return absTheta;}
		
		//! @brief Read the width of phi bins from the pre-constructed cache.
		virtual double PhiWidth(towerID_t const thetaIndex) const;
		
	private:
		// Each theta belt has a different number of phi bins, to maintain dOmega.
		// To implement the bijective towerID mapping quickly, 
		// we store the dPhi for each theta belt.
		std::vector<double> phiWidth; 
		
		void Init_phiWidth(); // Set up phiWidth
		
	public:
		ArrogantDetector_Lepton(QSettings const& parsedINI, 
			std::string const& detectorName = "detector"):
		ArrogantDetector(parsedINI, detectorName)
		{
			Init_inDerivedCTOR();
			Init_phiWidth();
		}
		virtual ~ArrogantDetector_Lepton() {}
};

/*! @brief An ArrogantDetector at a hadron collider.
 * 
 *  The hadron collider uses pseudorapidity \f$ \eta = {\tt arctanh}\left(\frac{p_L}{p}\right)\f$.
 *  Because the detector is cylindrical, every band has the same number of towers.
*/ 
class ArrogantDetector_Hadron : public ArrogantDetector
{
	protected:
		//! @brief \f$ \eta = {\tt arctanh}\left(\frac{p_L}{p}\right)\f$.
		virtual double AbsPolarAngle(vec3_t const& particle) const;
		
		//! @brief \f$ \theta = \tanh(\eta) \f$
		virtual double ToAbsTheta(double const absEta) const;
		
		//! @brief \a phi bins have constant width \p squareWidth
		virtual double PhiWidth(towerID_t const thetaIndex) const {return settings.squareWidth;}
		
	public:
		ArrogantDetector_Hadron(QSettings const& parsedINI, 
			std::string const& detectorName = "detector"):
		ArrogantDetector(parsedINI, detectorName)
		{
			Init_inDerivedCTOR();
		}
		virtual ~ArrogantDetector_Hadron() {}
};

#endif
